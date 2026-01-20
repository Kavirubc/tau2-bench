"""
Domain registry for multi-domain compensation benchmarking.

Provides centralized configuration and loading for all supported τ²-bench domains.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("tau2_integration.domain_registry")


@dataclass
class DomainConfig:
    """Configuration for a single domain."""
    
    name: str
    data_dir: Path
    db_path: Path
    tasks_path: Path
    policy_path: Path
    tools_module: str
    db_model_class: str
    toolkit_class: str
    
    def __post_init__(self):
        """Validate paths exist."""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        if not self.tasks_path.exists():
            raise ValueError(f"Tasks file not found: {self.tasks_path}")
        if not self.policy_path.exists():
            logger.warning(f"Policy file not found: {self.policy_path}")


class DomainRegistry:
    """
    Central registry for all supported τ²-bench domains.
    
    Manages domain configurations and provides utilities for loading
    domain-specific tools, policies, and tasks.
    """
    
    # Base paths
    _BASE_DIR = Path(__file__).parent.parent / "data" / "tau2" / "domains"
    
    # Domain configurations
    _DOMAINS: Dict[str, DomainConfig] = {}
    
    @classmethod
    def _initialize_domains(cls) -> None:
        """Initialize all domain configurations."""
        if cls._DOMAINS:
            return  # Already initialized
        
        # Airline domain
        airline_dir = cls._BASE_DIR / "airline"
        cls._DOMAINS["airline"] = DomainConfig(
            name="airline",
            data_dir=airline_dir,
            db_path=airline_dir / "db.json",
            tasks_path=airline_dir / "tasks.json",
            policy_path=airline_dir / "policy.md",
            tools_module="tau2.domains.airline.tools",
            db_model_class="tau2.domains.airline.data_model.FlightDB",
            toolkit_class="AirlineTools",
        )
        
        # Retail domain
        retail_dir = cls._BASE_DIR / "retail"
        cls._DOMAINS["retail"] = DomainConfig(
            name="retail",
            data_dir=retail_dir,
            db_path=retail_dir / "db.json",
            tasks_path=retail_dir / "tasks.json",
            policy_path=retail_dir / "policy.md",
            tools_module="tau2.domains.retail.tools",
            db_model_class="tau2.domains.retail.data_model.RetailDB",
            toolkit_class="RetailTools",
        )
        
        # Telecom domain
        telecom_dir = cls._BASE_DIR / "telecom"
        cls._DOMAINS["telecom"] = DomainConfig(
            name="telecom",
            data_dir=telecom_dir,
            db_path=telecom_dir / "db.toml",  # Note: TOML format
            tasks_path=telecom_dir / "tasks.json",
            policy_path=telecom_dir / "main_policy.md",
            tools_module="tau2.domains.telecom.tools",
            db_model_class="tau2.domains.telecom.data_model.TelecomDB",
            toolkit_class="TelecomTools",
        )
        
        # Mock domain
        mock_dir = cls._BASE_DIR / "mock"
        cls._DOMAINS["mock"] = DomainConfig(
            name="mock",
            data_dir=mock_dir,
            db_path=mock_dir / "db.json",
            tasks_path=mock_dir / "tasks.json",
            policy_path=mock_dir / "policy.md",
            tools_module="tau2.domains.mock.tools",
            db_model_class="tau2.domains.mock.data_model.MockDB",
            toolkit_class="MockTools",
        )
        
        logger.info(f"Initialized {len(cls._DOMAINS)} domains: {list(cls._DOMAINS.keys())}")
    
    @classmethod
    def get_domain(cls, name: str) -> DomainConfig:
        """
        Get configuration for a domain.
        
        Args:
            name: Domain name (airline, retail, telecom, mock)
            
        Returns:
            DomainConfig for the specified domain
            
        Raises:
            ValueError: If domain is not supported
        """
        cls._initialize_domains()
        
        if name not in cls._DOMAINS:
            raise ValueError(
                f"Unknown domain: {name}. "
                f"Supported domains: {list(cls._DOMAINS.keys())}"
            )
        
        return cls._DOMAINS[name]
    
    @classmethod
    def list_domains(cls) -> List[str]:
        """
        List all available domains.
        
        Returns:
            List of domain names
        """
        cls._initialize_domains()
        return list(cls._DOMAINS.keys())
    
    @classmethod
    def load_domain_tools(cls, name: str) -> Dict[str, Callable]:
        """
        Load tools for a domain.
        
        Args:
            name: Domain name
            
        Returns:
            Dictionary mapping tool names to tool functions
            
        Raises:
            ImportError: If domain modules cannot be imported
            ValueError: If domain is not supported
        """
        config = cls.get_domain(name)
        
        try:
            # Import the tools module
            import importlib
            tools_module = importlib.import_module(config.tools_module)
            
            # Import the DB model class
            db_module_name, db_class_name = config.db_model_class.rsplit(".", 1)
            db_module = importlib.import_module(db_module_name)
            db_class = getattr(db_module, db_class_name)
            
            # Load the database
            db = db_class.load(config.db_path)
            
            # Get the toolkit class
            toolkit_class = getattr(tools_module, config.toolkit_class)
            
            # Instantiate the toolkit
            toolkit = toolkit_class(db)
            
            # Extract all tool functions
            tools = {}
            for name in dir(toolkit):
                if name.startswith("_"):
                    continue
                attr = getattr(toolkit, name)
                if callable(attr) and hasattr(attr, "__self__"):
                    tools[name] = attr
            
            logger.info(f"Loaded {len(tools)} tools for {config.name} domain")
            return tools
            
        except ImportError as e:
            logger.error(f"Failed to import {config.name} domain: {e}")
            raise ImportError(
                f"Failed to load {config.name} domain. "
                f"Make sure τ²-bench is installed: pip install -e ."
            ) from e
    
    @classmethod
    def load_domain_policy(cls, name: str) -> str:
        """
        Load policy document for a domain.
        
        Args:
            name: Domain name
            
        Returns:
            Policy text as string
            
        Raises:
            ValueError: If domain is not supported
            FileNotFoundError: If policy file doesn't exist
        """
        config = cls.get_domain(name)
        
        if not config.policy_path.exists():
            logger.warning(f"Policy file not found for {name} domain")
            return f"No policy found for {name} domain."
        
        policy_text = config.policy_path.read_text()
        logger.info(f"Loaded policy for {name} domain ({len(policy_text)} chars)")
        return policy_text
    
    @classmethod
    def get_domain_db_path(cls, name: str) -> Path:
        """
        Get database path for a domain.
        
        Args:
            name: Domain name
            
        Returns:
            Path to domain database file
        """
        config = cls.get_domain(name)
        return config.db_path
    
    @classmethod
    def get_domain_tasks_path(cls, name: str) -> Path:
        """
        Get tasks file path for a domain.
        
        Args:
            name: Domain name
            
        Returns:
            Path to domain tasks.json file
        """
        config = cls.get_domain(name)
        return config.tasks_path


# Convenience function
def get_domain_config(name: str) -> DomainConfig:
    """Get domain configuration. Alias for DomainRegistry.get_domain()."""
    return DomainRegistry.get_domain(name)

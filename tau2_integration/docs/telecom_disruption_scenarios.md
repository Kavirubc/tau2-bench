# Telecom Domain - Disruption Scenarios Analysis

## Tool Inventory

### READ Tools (Query Operations)
- `get_customer_by_phone` - Find customer by phone number
- `get_customer_by_id` - Get customer by ID
- `get_customer_by_name` - Search customers by name and DOB
- `get_details_by_id` - Get details for any ID (Customer, Line, Device, Bill, Plan)
- `get_bills_for_customer` - Retrieve customer bills
- `get_data_usage` - Get data usage for a line

### WRITE Tools (State-Modifying Operations)
- `suspend_line` - Suspend a line (max 6 months)
- `resume_line` - Resume a suspended line
- `send_payment_request` - Send payment request for a bill
- `enable_roaming` - Enable international roaming
- `disable_roaming` - Disable international roaming
- `refuel_data` - Add data to a line (charges customer)

### Internal/Helper Methods (Not exposed as tools)
- `set_data_usage` - Set data usage (testing)
- `suspend_line_for_overdue_bill` - System suspension for non-payment
- `_set_bill_to_paid` - Mark bill as paid
- `_apply_one_time_charge` - Add charge to bill

### GENERIC Tools
- `transfer_to_human_agents` - Escalate to human

## Compensation Pairs

| Action Tool | Compensation Tool | Notes |
|-------------|-------------------|-------|
| `suspend_line` | `resume_line` | Direct inverse operation |
| `resume_line` | `suspend_line` | Can re-suspend if needed |
| `enable_roaming` | `disable_roaming` | Direct inverse operation |
| `disable_roaming` | `enable_roaming` | Direct inverse operation |
| `refuel_data` | N/A | Cannot remove data once added (charge applied) |
| `send_payment_request` | N/A | Cannot unsend payment request |

## Disruption Scenarios

### 1. Transient Failures (Retry May Succeed)

#### Billing System Timeout
- **Type**: `billing_system_timeout`
- **Affected Tools**: `send_payment_request`, `refuel_data`, `get_bills_for_customer`
- **Trigger**: After 2 actions
- **Persistent**: No
- **Retries Until Success**: 2
- **Message**: "Billing system temporarily unavailable - please retry"
- **Expected Behavior**: Agent should retry the billing operation

#### Network API Delay
- **Type**: `network_api_delay`
- **Affected Tools**: `suspend_line`, `resume_line`, `enable_roaming`, `disable_roaming`
- **Trigger**: After 3 actions
- **Persistent**: No
- **Retries Until Success**: 1
- **Message**: "Network configuration API timeout - retry in a moment"
- **Expected Behavior**: Agent should wait and retry network operations

#### Customer Database Lock
- **Type**: `customer_db_lock`
- **Affected Tools**: `get_customer_by_phone`, `get_customer_by_id`, `get_customer_by_name`
- **Trigger**: After 2 actions
- **Persistent**: No
- **Retries Until Success**: 1
- **Message**: "Customer database temporarily locked - please retry"
- **Expected Behavior**: Agent should retry customer lookup

### 2. Persistent Failures (Requires Replanning)

#### Service Plan Unavailable
- **Type**: `service_plan_unavailable`
- **Affected Tools**: `get_details_by_id` (when querying plans)
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "Service plan {plan_id} is no longer available for new customers"
- **Expected Behavior**: Agent should suggest alternative plans

#### Account Suspended
- **Type**: `account_suspended`
- **Affected Tools**: `refuel_data`, `enable_roaming`, `send_payment_request`
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "Account is suspended due to non-payment - clear outstanding balance first"
- **Expected Behavior**: Agent should guide user to pay outstanding bills

#### Line Not Found
- **Type**: `line_not_found`
- **Affected Tools**: `suspend_line`, `resume_line`, `get_data_usage`, `refuel_data`
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "Line {line_id} not found for customer {customer_id}"
- **Expected Behavior**: Agent should verify line ID with user

#### Line Already Suspended
- **Type**: `line_already_suspended`
- **Affected Tools**: `suspend_line`
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "Line is already suspended - cannot suspend again"
- **Expected Behavior**: Agent should inform user of current status

#### Line Not Suspended
- **Type**: `line_not_suspended`
- **Affected Tools**: `resume_line`
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "Line is not suspended - cannot resume"
- **Expected Behavior**: Agent should inform user line is already active

### 3. Complex Scenarios

#### Payment Already Pending
- **Type**: `payment_already_pending`
- **Affected Tools**: `send_payment_request`
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "A bill is already awaiting payment for this customer"
- **Expected Behavior**: Agent should inform user to complete pending payment first

#### Data Refuel Limit Exceeded
- **Type**: `data_refuel_limit_exceeded`
- **Affected Tools**: `refuel_data`
- **Trigger**: After 2 actions
- **Persistent**: Yes
- **Message**: "Data refuel limit exceeded for this billing cycle - max 50GB"
- **Expected Behavior**: Agent should inform user of limit and suggest plan upgrade

#### Roaming Already Enabled/Disabled
- **Type**: `roaming_state_conflict`
- **Affected Tools**: `enable_roaming`, `disable_roaming`
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "Roaming was already {enabled/disabled}"
- **Expected Behavior**: Agent should acknowledge current state

## Disruption Injection Strategy

### For Compensation Testing

1. **Scenario 1**: Billing timeout during data refuel
   - Refuel data → Billing timeout → Retry → Success
   - Tests: Transient failure handling, retry logic, charge application

2. **Scenario 2**: Account suspended during roaming enable
   - Enable roaming → Account suspended → Guide to payment → Resume after payment
   - Tests: Persistent failure, multi-step recovery, compensation

3. **Scenario 3**: Line suspension with network failure
   - Suspend line → Network API delay → Retry → Success
   - Tests: Transient network failure, retry persistence

4. **Scenario 4**: Data refuel with cascading failures
   - Refuel data → Customer DB lock → Retry → Billing timeout → Retry → Success
   - Tests: Multiple transient failures, retry logic

5. **Scenario 5**: Payment request with existing pending payment
   - Send payment request → Already pending → Inform user → Wait for completion
   - Tests: Persistent failure, state validation

## Expected Compensation Behaviors

### SagaLLM
- Should detect billing timeouts and retry
- Should detect persistent failures (account suspended) and trigger compensation
- Should reverse line suspension if downstream operations fail
- Should handle multi-step processes (suspend → charge → resume)

### RAC (React Agent Compensation)
- Should automatically retry transient failures (billing, network)
- Should detect tool output errors (already suspended, not found)
- Should trigger compensation for failed state changes (resume after failed suspend)

### Vanilla LangGraph
- Relies on LLM reasoning to handle errors
- May or may not retry depending on prompt engineering
- No automatic compensation mechanism
- May struggle with complex multi-step failures

## Test Task Examples

### Task 1: Simple Line Suspension with Transient Failure
- User wants to suspend line for travel
- Network API times out twice, succeeds on third try
- Expected: All frameworks should eventually succeed

### Task 2: Data Refuel with Account Issue
- User wants to add data
- Account is suspended due to non-payment
- Agent must guide to payment first, then refuel
- Expected: SagaLLM and RAC should handle gracefully, LangGraph may struggle

### Task 3: Roaming Enable with Billing Failure
- User wants to enable roaming before travel
- Billing system timeout during charge application
- Agent must retry and confirm success
- Expected: SagaLLM should compensate if partial failure, RAC should retry

### Task 4: Complex Multi-Step with Cascading Failures
- User wants to suspend line, refuel data on another line, enable roaming
- Multiple transient failures across operations
- Expected: SagaLLM should maintain transaction integrity, RAC should retry each, LangGraph may fail

## Domain-Specific Considerations

### Billing Integrity
- Data refuel charges must be applied correctly or not at all
- Cannot partially refuel data (all-or-nothing operation)
- Compensation must handle charge reversal if operation fails

### Line State Management
- Line status changes must be atomic (suspended ↔ active)
- Cannot have intermediate states
- Compensation must restore previous state if operation fails

### Customer Account State
- Account suspension affects all operations
- Must check account state before any write operation
- Compensation may require account-level recovery

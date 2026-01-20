# Mock Domain - Disruption Scenarios Analysis

## Tool Inventory

### READ Tools (Query Operations)
- `get_users` - Get all users in the database

### WRITE Tools (State-Modifying Operations)
- `create_task` - Create a new task for a user
- `update_task_status` - Update the status of a task

### GENERIC Tools
- `transfer_to_human_agents` - Escalate to human

### Assertion Methods (Testing)
- `assert_number_of_tasks` - Verify task count for user
- `assert_task_status` - Verify task status

## Compensation Pairs

| Action Tool | Compensation Tool | Notes |
|-------------|-------------------|-------|
| `create_task` | N/A | Cannot delete tasks (no delete tool) |
| `update_task_status` | `update_task_status` (reverse) | Can revert to previous status |

## Disruption Scenarios

### 1. Transient Failures (Retry May Succeed)

#### Database Lock
- **Type**: `database_lock`
- **Affected Tools**: `create_task`, `update_task_status`
- **Trigger**: After 2 actions
- **Persistent**: No
- **Retries Until Success**: 1
- **Message**: "Database temporarily locked - please retry"
- **Expected Behavior**: Agent should retry the operation

#### API Timeout
- **Type**: `api_timeout`
- **Affected Tools**: `get_users`, `create_task`, `update_task_status`
- **Trigger**: After 3 actions
- **Persistent**: No
- **Retries Until Success**: 2
- **Message**: "API request timeout - please retry"
- **Expected Behavior**: Agent should wait and retry

#### Connection Pool Exhausted
- **Type**: `connection_pool_exhausted`
- **Affected Tools**: Any tool
- **Trigger**: After 2 actions
- **Persistent**: No
- **Retries Until Success**: 1
- **Message**: "Connection pool exhausted - retry shortly"
- **Expected Behavior**: Agent should retry after brief delay

### 2. Persistent Failures (Requires Replanning)

#### Task Not Found
- **Type**: `task_not_found`
- **Affected Tools**: `update_task_status`
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "Task {task_id} not found in system"
- **Expected Behavior**: Agent should verify task ID with user

#### User Not Found
- **Type**: `user_not_found`
- **Affected Tools**: `create_task`
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "User {user_id} not found"
- **Expected Behavior**: Agent should verify user ID or create user first

#### User Not Authorized
- **Type**: `user_not_authorized`
- **Affected Tools**: `update_task_status`, `create_task`
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "User {user_id} not authorized to perform this action"
- **Expected Behavior**: Agent should inform user of permission issue

#### Invalid Status Transition
- **Type**: `invalid_status_transition`
- **Affected Tools**: `update_task_status`
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "Cannot transition task from {current_status} to {new_status}"
- **Expected Behavior**: Agent should inform user of valid transitions

### 3. Complex Scenarios

#### Task Limit Exceeded
- **Type**: `task_limit_exceeded`
- **Affected Tools**: `create_task`
- **Trigger**: After 2 actions
- **Persistent**: Yes
- **Message**: "User has reached maximum task limit (100 tasks)"
- **Expected Behavior**: Agent should suggest completing existing tasks first

#### Duplicate Task Title
- **Type**: `duplicate_task_title`
- **Affected Tools**: `create_task`
- **Trigger**: After 1 action
- **Persistent**: No
- **Retries Until Success**: 0 (requires different input)
- **Message**: "Task with title '{title}' already exists for this user"
- **Expected Behavior**: Agent should suggest modifying title

## Disruption Injection Strategy

### For Compensation Testing

1. **Scenario 1**: Database lock during task creation
   - Create task → Database lock → Retry → Success
   - Tests: Transient failure handling, retry logic

2. **Scenario 2**: Task not found during status update
   - Update status → Task not found → Verify ID → Retry with correct ID
   - Tests: Persistent failure, error recovery

3. **Scenario 3**: User not authorized
   - Create task → Not authorized → Inform user → Transfer to human
   - Tests: Persistent failure, escalation

4. **Scenario 4**: Multiple failures cascade
   - Create task → API timeout → Retry → Database lock → Retry → Success
   - Tests: Multiple transient failures, retry persistence

5. **Scenario 5**: Status update with invalid transition
   - Update status → Invalid transition → Inform user → Update to valid status
   - Tests: Persistent failure, alternative solution

## Expected Compensation Behaviors

### SagaLLM
- Should detect database locks and retry
- Should detect persistent failures (not found, not authorized) and trigger compensation
- Should reverse task creation if downstream operations fail
- Should handle status transitions atomically

### RAC (React Agent Compensation)
- Should automatically retry transient failures (database lock, API timeout)
- Should detect tool output errors (not found, not authorized)
- Should trigger compensation for failed operations

### Vanilla LangGraph
- Relies on LLM reasoning to handle errors
- May or may not retry depending on prompt
- No automatic compensation mechanism
- May struggle with complex error scenarios

## Test Task Examples

### Task 1: Simple Task Creation with Transient Failure
- User wants to create a task
- Database lock occurs twice, succeeds on third try
- Expected: All frameworks should eventually succeed

### Task 2: Task Update with Not Found Error
- User wants to update task status
- Task ID is incorrect/not found
- Agent must verify ID with user
- Expected: SagaLLM and RAC should handle gracefully, LangGraph may struggle

### Task 3: Task Creation with Authorization Issue
- User wants to create task
- User not authorized
- Agent must inform user and potentially escalate
- Expected: All frameworks should inform user, SagaLLM/RAC may handle better

### Task 4: Cascading Failures
- Create task → API timeout → Retry → Update status → Database lock → Retry → Success
- Expected: SagaLLM should maintain integrity, RAC should retry, LangGraph may fail

## Domain-Specific Considerations

### Simplicity
- Mock domain has minimal tools (only 2 write operations)
- Easier to test basic compensation patterns
- Good baseline for framework comparison

### No Delete Operation
- Cannot compensate task creation by deletion
- Must rely on status updates for compensation
- Tests framework handling of irreversible operations

### Status State Machine
- Task status follows a state machine (pending → in_progress → completed)
- Invalid transitions should be caught
- Compensation must respect state machine rules

### Minimal Side Effects
- Operations have minimal side effects
- Easier to reason about compensation
- Good for testing core compensation logic without complex domain rules

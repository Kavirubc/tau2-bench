# Retail Domain - Disruption Scenarios Analysis

## Tool Inventory

### READ Tools (Query Operations)
- `find_user_id_by_email` - Find user by email
- `find_user_id_by_name_zip` - Find user by name and zip
- `get_order_details` - Get order status and details
- `get_product_details` - Get product inventory
- `get_user_details` - Get user information
- `list_all_product_types` - List all products
- `calculate` - Mathematical calculations

### WRITE Tools (State-Modifying Operations)
- `cancel_pending_order` - Cancel pending order (requires confirmation)
- `exchange_delivered_order_items` - Exchange items in delivered order
- `modify_pending_order_address` - Modify shipping address
- `modify_pending_order_items` - Modify items in pending order
- `modify_pending_order_payment` - Change payment method
- `modify_user_address` - Update user's default address
- `return_delivered_order_items` - Return items from delivered order

### GENERIC Tools
- `transfer_to_human_agents` - Escalate to human

## Compensation Pairs

| Action Tool | Compensation Tool | Notes |
|-------------|-------------------|-------|
| `modify_pending_order_items` | `modify_pending_order_items` (reverse) | Swap back to original items |
| `modify_pending_order_payment` | `modify_pending_order_payment` (reverse) | Revert to original payment method |
| `modify_pending_order_address` | `modify_pending_order_address` (reverse) | Restore original address |
| `modify_user_address` | `modify_user_address` (reverse) | Restore previous address |
| `exchange_delivered_order_items` | N/A | Cannot undo once requested |
| `return_delivered_order_items` | N/A | Cannot undo once requested |
| `cancel_pending_order` | N/A | Irreversible action |

## Disruption Scenarios

### 1. Transient Failures (Retry May Succeed)

#### Payment Gateway Timeout
- **Type**: `payment_gateway_timeout`
- **Affected Tools**: `modify_pending_order_payment`, `modify_pending_order_items` (when price diff)
- **Trigger**: After 2 actions
- **Persistent**: No
- **Retries Until Success**: 2
- **Message**: "Payment gateway timeout - please retry in a moment"
- **Expected Behavior**: Agent should retry the operation

#### Inventory Sync Delay
- **Type**: `inventory_sync_delay`
- **Affected Tools**: `get_product_details`, `modify_pending_order_items`
- **Trigger**: After 3 actions
- **Persistent**: No
- **Retries Until Success**: 1
- **Message**: "Inventory system temporarily unavailable - retry shortly"
- **Expected Behavior**: Agent should wait and retry

#### Database Lock
- **Type**: `database_lock`
- **Affected Tools**: Any WRITE tool
- **Trigger**: After 2 actions
- **Persistent**: No
- **Retries Until Success**: 1
- **Message**: "Database temporarily locked - please retry"
- **Expected Behavior**: Agent should retry after brief delay

### 2. Persistent Failures (Requires Replanning)

#### Product Out of Stock
- **Type**: `product_out_of_stock`
- **Affected Tools**: `modify_pending_order_items`, `exchange_delivered_order_items`
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "Product variant {variant_id} is currently out of stock"
- **Expected Behavior**: Agent should find alternative product or inform user

#### Invalid Product ID
- **Type**: `invalid_product_id`
- **Affected Tools**: `get_product_details`, `modify_pending_order_items`
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "Product ID {product_id} not found in catalog"
- **Expected Behavior**: Agent should verify product ID with user

#### Insufficient Gift Card Balance
- **Type**: `insufficient_balance`
- **Affected Tools**: `modify_pending_order_payment`, `modify_pending_order_items`
- **Trigger**: After 2 actions
- **Persistent**: Yes
- **Message**: "Insufficient gift card balance to complete transaction"
- **Expected Behavior**: Agent should suggest alternative payment method

#### Order Already Processed
- **Type**: `order_already_processed`
- **Affected Tools**: `cancel_pending_order`, `modify_pending_order_*`
- **Trigger**: After 1 action
- **Persistent**: Yes
- **Message**: "Order has already been processed and cannot be modified"
- **Expected Behavior**: Agent should inform user and suggest alternatives

### 3. Complex Scenarios

#### Warehouse Unavailable
- **Type**: `warehouse_unavailable`
- **Affected Tools**: `exchange_delivered_order_items`, `return_delivered_order_items`
- **Trigger**: After 2 actions
- **Persistent**: Yes
- **Message**: "Warehouse for your region is temporarily closed - returns/exchanges unavailable"
- **Expected Behavior**: Agent should inform user of delay and provide timeline

#### Address Validation Failure
- **Type**: `address_validation_failure`
- **Affected Tools**: `modify_pending_order_address`, `modify_user_address`
- **Trigger**: After 1 action
- **Persistent**: No
- **Retries Until Success**: 2
- **Message**: "Address validation service unavailable - please retry"
- **Expected Behavior**: Agent should retry or ask user to verify address

## Disruption Injection Strategy

### For Compensation Testing
1. **Scenario 1**: Payment timeout during order modification
   - Modify items → Payment timeout → Retry → Success
   - Tests: Transient failure handling, retry logic

2. **Scenario 2**: Product out of stock during exchange
   - Exchange items → Product unavailable → Find alternative → Complete with different item
   - Tests: Persistent failure, replanning, compensation

3. **Scenario 3**: Order processed during cancellation attempt
   - Cancel order → Already processed → Inform user → Suggest return instead
   - Tests: Persistent failure, alternative solution

4. **Scenario 4**: Multiple failures cascade
   - Modify items → Inventory delay → Retry → Payment timeout → Retry → Success
   - Tests: Multiple transient failures, retry persistence

## Expected Compensation Behaviors

### SagaLLM
- Should detect payment timeout and retry
- Should detect persistent failures and trigger compensation
- Should reverse partial modifications if downstream fails

### RAC (React Agent Compensation)
- Should automatically retry transient failures
- Should detect tool output errors
- Should trigger compensation actions for failed operations

### Vanilla LangGraph
- Relies on LLM reasoning to handle errors
- May or may not retry depending on prompt
- No automatic compensation mechanism

## Test Task Examples

### Task 1: Simple Modification with Transient Failure
- User wants to change order items
- Payment gateway times out twice, succeeds on third try
- Expected: All frameworks should eventually succeed

### Task 2: Modification with Persistent Failure
- User wants to exchange to out-of-stock item
- Agent must find alternative or inform user
- Expected: SagaLLM and RAC should handle gracefully, LangGraph may struggle

### Task 3: Cascading Failures
- Multiple operations with mixed transient/persistent failures
- Expected: SagaLLM should compensate properly, RAC should retry, LangGraph may fail

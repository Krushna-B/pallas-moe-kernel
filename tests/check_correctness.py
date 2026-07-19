import jax
import jax.numpy as jnp

from src.kernel import fused_ragged_dot_caller

# Sample testing to ensure correctness of kernel operations
key = jax.random.PRNGKey(0)
M = 2048  # Total routed tokens
E = 4  # Number of experts
D = 512  # Model dimension
F = 2048  # Expert hidden dimension
BM = 128
BF = 32
group_sizes = jnp.array(
    [470, 530, 421, 627],
    dtype=jnp.int32,
)
assert int(jnp.sum(group_sizes)) == M

max_group_size = max(
    [470, 530, 421, 627],
)

k1, k2, k3 = jax.random.split(key, 3)
# Build input matricies
x = jax.random.normal(
    k1,
    (M, D),
    dtype=jnp.float32,
)

# One up/down projection per expert
w_up = jax.random.normal(
    k2,
    (E, D, F),
    dtype=jnp.float32,
)

w_down = jax.random.normal(
    k3,
    (E, F, D),
    dtype=jnp.float32,
)
y_pallas = fused_ragged_dot_caller(
    x,
    w_up,
    w_down,
    group_sizes,
    max_group_size=max_group_size,
    BM=BM,
    BF=BF,
    interpret=True,
)

# Actual ragged dot values
# (M, D) ragged-dot (E, D, F) -> (M, F)
h_ref = jax.lax.ragged_dot(
    x,
    w_up,
    group_sizes,
)

# Important: use the same GELU approximation as the kernel.
h_ref = jax.nn.gelu(
    h_ref,
)

# (M, F) ragged-dot (E, F, D) -> (M, D)
y_ref = jax.lax.ragged_dot(
    h_ref,
    w_down,
    group_sizes,
)
y_pallas.block_until_ready()
y_ref.block_until_ready()

### Metrics
difference = y_pallas - y_ref
absolute_error = jnp.abs(difference)

max_absolute_error = jnp.max(absolute_error)
mean_absolute_error = jnp.mean(absolute_error)

# Pointwise relative error can become large when y_ref is near zero,
# so clamp the denominator
relative_error = absolute_error / jnp.maximum(
    jnp.abs(y_ref),
    1e-6,
)

max_relative_error = jnp.max(relative_error)
mean_relative_error = jnp.mean(relative_error)


l2_relative_error = jnp.linalg.norm(difference) / jnp.maximum(
    jnp.linalg.norm(y_ref), 1e-6
)

is_close = jnp.allclose(
    y_pallas,
    y_ref,
    rtol=1e-3,
    atol=1e-3,
)


print("Pallas output shape:", y_pallas.shape)
print("Reference output shape:", y_ref.shape)
print()

print("Maximum absolute error:", float(max_absolute_error))
print("Mean absolute error:   ", float(mean_absolute_error))
print()

print("Maximum relative error:", float(max_relative_error))
print("Mean relative error:   ", float(mean_relative_error))
print("L2 relative error:     ", float(l2_relative_error))
print()

print("Outputs close:", bool(is_close))


diff = y_pallas - y_ref
abs_diff = jnp.abs(diff)

# Largest absolute-error location
max_abs_flat = jnp.argmax(abs_diff)
max_abs_idx = jnp.unravel_index(max_abs_flat, abs_diff.shape)

print("\nWorst absolute-error entry")
print("Index:       ", tuple(int(i) for i in max_abs_idx))
print("Pallas:     ", float(y_pallas[max_abs_idx]))
print("Reference:  ", float(y_ref[max_abs_idx]))
print("Abs error:  ", float(abs_diff[max_abs_idx]))
print(
    "Rel error:  ",
    float(abs_diff[max_abs_idx] / jnp.maximum(jnp.abs(y_ref[max_abs_idx]), 1e-12)),
)

# Largest relative-error location
pointwise_rel = abs_diff / jnp.maximum(jnp.abs(y_ref), 1e-12)

max_rel_flat = jnp.argmax(pointwise_rel)
max_rel_idx = jnp.unravel_index(max_rel_flat, pointwise_rel.shape)

print("\nWorst relative-error entry")
print("Index:       ", tuple(int(i) for i in max_rel_idx))
print("Pallas:     ", float(y_pallas[max_rel_idx]))
print("Reference:  ", float(y_ref[max_rel_idx]))
print("Abs error:  ", float(abs_diff[max_rel_idx]))
print("Rel error:  ", float(pointwise_rel[max_rel_idx]))

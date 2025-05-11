## LESSON:
torch.Tensor.long --> mx.array.astype(mx.int64)

## LESSON:
ARRAY.expand(3, -1) --> mx.repeat(ARRAY, repeats=3, axis=0)
ARRAY.expand(3, -1, -1) --> mx.repeat(ARRAY, repeats=3, axis=0)


Torch:
```
torch.Tensor.expand
Tensor.expand(*sizes) → Tensor
Returns a new view of the self tensor with singleton dimensions expanded to a larger size.

Passing -1 as the size for a dimension means not changing the size of that dimension.

Tensor can be also expanded to a larger number of dimensions, and the new ones will be appended at the front. For the new dimensions, the size cannot be set to -1.

Expanding a tensor does not allocate new memory, but only creates a new view on the existing tensor where a dimension of size one is expanded to a larger size by setting the stride to 0. Any dimension of size 1 can be expanded to an arbitrary value without allocating new memory.

Parameters
*sizes (torch.Size or int...) – the desired expanded size
```

MLX:
```
mlx.core.repeat
repeat(array: array, repeats: int, axis: int | None = None, *, stream: None | Stream | Device = None) → array
Repeat an array along a specified axis.

Parameters
:
array (array) – Input array.

repeats (int) – The number of repetitions for each element.

axis (int, optional) – The axis in which to repeat the array along. If unspecified it uses the flattened array of the input and repeats along axis 0.

stream (Stream, optional) – Stream or device. Defaults to None.

Returns
:
The resulting repeated array.

Return type
:
array
```

### EXAMPLE:
TORCH:
h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten()

MLX:
h_index = mx.arange(int(llm_grid_h)).reshape(1, -1, 1)
h_index = mx.repeat(h_index, repeats=len(t_index), axis=0)
h_index = mx.repeat(h_index, repeats=llm_grid_w, axis=2)
h_index = h_index.flatten()


## LESSON:
torch.arange(text_len) --> mx.arange(int(text_len))

## LESSON:
torch.cat(llm_pos_ids_list, dim=1) --> mx.concatenate(llm_pos_ids_list, axis=1)

## LESSON:
ARRAY.unsqueeze(X) --> mx.expand_dims(ARRAY, X)

## LESSON:
ARRAY.float() --> ARRAY.astype(mx.float32)

## LESSON:
torch.empty((X, Y), dtype=torch.int64) --> mx.random.randint(-1000000000, 1000000000, (X, Y), dtype=mx.int64)

## LESSON:
Remove all device moving, it's not needed for MLX
`.to(ARRAY.device)`
`.to(torch_device)`

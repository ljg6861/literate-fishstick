import numpy as np

def original_logic(local_start, local_end, puzzle_indices_arr):
    puzzle_indices = []
    puzzle_index = np.searchsorted(puzzle_indices_arr, local_start, side="right") - 1
    for i in range(local_start, local_end):
        while puzzle_index + 1 < len(puzzle_indices_arr) and i >= puzzle_indices_arr[puzzle_index + 1]:
            puzzle_index += 1
        puzzle_indices.append(puzzle_index)
    return np.array(puzzle_indices)

def optimized_logic(local_start, local_end, puzzle_indices_arr):
    indices = np.arange(local_start, local_end)
    puzzle_indices = np.searchsorted(puzzle_indices_arr, indices, side="right") - 1
    return puzzle_indices

def test_logic_equivalence():
    num_puzzles = 100
    lengths = np.random.randint(5, 50, size=num_puzzles)
    puzzle_indices_arr = np.concatenate(([0], np.cumsum(lengths)))
    total_examples = puzzle_indices_arr[-1]

    batch_size = 32

    for _ in range(50):
        start_idx = np.random.randint(0, total_examples - batch_size)
        end_idx = start_idx + batch_size

        expected = original_logic(start_idx, end_idx, puzzle_indices_arr)
        actual = optimized_logic(start_idx, end_idx, puzzle_indices_arr)

        np.testing.assert_array_equal(expected, actual)

if __name__ == "__main__":
    try:
        test_logic_equivalence()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        exit(1)

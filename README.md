# Makemore - Character-level Language Model


## Files in this Repository

open the 'makemore' directory and you'll find:
    - `makemore.ipynb`: The jupyter notebook in the video
    - `my_makemore_template.py`: A template file with function interfaces and documentation, but implementation left for the user to complete
    - `my_makemore_solution.py`: A complete implementation of the character-level language model
    - `names.txt`: The dataset containing names used for training (make sure to include this file)

## How to Use

### Prerequisites

- Python 3.6+
- PyTorch
- Matplotlib (for plotting training loss)

### Running the Model

1. Clone this repository
2. Ensure you have the required dependencies installed
3. Run the solution file:

```bash
python my_makemore_solution.py
```

This will:
- Load the names dataset
- Build the vocabulary
- Train the model
- Evaluate its performance
- Generate sample names

### Implementing Your Own Version

If you want to implement the model yourself:

1. Start with `my_makemore_template.py`
2. Fill in the sections marked with "your code here"
3. Run your implementation to see how it performs

## Model Performance

After training, the model typically achieves:
- Training loss: ~1.77
- Validation loss: ~1.99

The model can generate names like:
- Arlij
- Chetta
- Hendrix
- Jamylie
- Marianah

## License

This project is open source and available under the [MIT License](LICENSE).

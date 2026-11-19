Note:

- Each matrix has shape (output width, input width), and each bias vector matches its output width; adjacent widths match.
- computing precision, convert inputs to float32 at the very first.
- use torch.tanh
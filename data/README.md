# Data for pre-training

Huggingface dataset: [cats vs dogs](https://huggingface.co/datasets/cats_vs_dogs)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = preprocess(load_dataset("cats_vs_dogs")['train'], device = device)
dataset = SimClrData(huggingface_dataset=dataset)
dataloder = DataLoader(dataset, batch_size=10, shuffle = True)
```
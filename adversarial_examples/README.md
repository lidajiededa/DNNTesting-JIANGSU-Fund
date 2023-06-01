我们的对抗样本使用 [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) 生成

```python
import torchattacks

atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)

# Save
atk.save(data_loader, save_path="./AE.pt", verbose=True)
  
# Load
adv_loader = atk.load(load_path="./AE.pt")
```
# Transformer 구현 해보기
## Psudo-code
### import
```
import torch, torch.nn, np, plt
```
### hyper-params
```
input, hidden, output size
num_epochs, batch_size
learning_rate
```
### data
data <- dataloader
X, Y
x_train, y_train
x_test, y_test

### model prototype
model(input) -> output

### loss and optimizer
loss(pred, label)
grad(Adam, lr)  

### train
for epoch in range(num_epochs):
    forward prop: y_pred = model(x_train), loss(y_pred, y_train)
    back prop: optimizer(loss, grad)
    print and save if epoch % 10 == (10-1)

### test
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

### save chkpt

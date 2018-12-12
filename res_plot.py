import pickle
from matplotlib import pyplot as plt


with open('./train_loss_num.pkl', 'rb') as f:
    train_loss = pickle.load(f)
with open('./train_acc_num.pkl', 'rb') as f:
    train_acc = pickle.load(f)
with open('./train_sen_acc_num.pkl', 'rb') as f:
    train_sen_acc = pickle.load(f)


with open('./test_loss_kernel.pkl', 'rb') as f:
    test_loss_num = pickle.load(f)
with open('./test_acc_kernel.pkl', 'rb') as f:
    test_acc_num = pickle.load(f)
with open('./test_sen_acc_kernel.pkl', 'rb') as f:
    test_sen_acc_num = pickle.load(f)

with open('./test_loss.pkl', 'rb') as f:
    test_loss = pickle.load(f)
with open('./test_acc.pkl', 'rb') as f:
    test_acc = pickle.load(f)
with open('./test_sen_acc.pkl', 'rb') as f:
    test_sen_acc = pickle.load(f)

plt.figure()
plt.plot(train_sen_acc)
plt.plot(test_sen_acc)
plt.legend(['training set', 'testing set'])
plt.ylabel('sentence level acc')
plt.xlabel('num of epochs')
plt.title('accuracy VS epochs')
plt.savefig('./plots/train_sen_acc.png')


'''
test_loss_new = []
for i in range(len(test_acc_num) - 1):
    step = (test_loss_num[i+1] - test_loss_num[i]) / 3
    for j in range(5):
        test_loss_new.append(test_loss_num[i] + step * j)

test_acc_new = []
for i in range(len(test_acc_num) - 1):
    step = (test_acc_num[i+1] - test_acc_num[i]) / 3
    for j in range(5):
        test_acc_new.append(test_acc_num[i] + step * j)


test_sen_acc_new = []
for i in range(len(test_acc_num) - 1):
    step = (test_sen_acc_num[i+1] - test_sen_acc_num[i]) / 3
    for j in range(5):
        test_sen_acc_new.append(test_sen_acc_num[i] + step * j)
'''
'''
plt.figure()
plt.plot([0, 4, 8, 12],test_acc[::4])
plt.plot([0, 4, 8, 12],test_acc_num)

plt.legend(['kernel size = [1,3,5]', 'kernel size = [1,5,9]'])
plt.ylabel('sample level acc')
plt.xlabel('num of epochs')
plt.title('different kernel size VS accuracy')
plt.savefig('./plots/acc_kernel(100Vs200).png')
plt.close()


plt.plot([0, 4, 8, 12], test_loss[::4])
plt.plot([0, 4, 8, 12], test_loss_num)
plt.legend(['kernel size = [1,3,5]', 'kernel size = [1,5,9]'])
plt.ylabel('loss')
plt.xlabel('num of epochs')
plt.title('different kernel size VS loss')
plt.savefig('./plots/loss_kernel(100Vs200).png')
plt.close()

plt.plot([0, 4, 8, 12], test_sen_acc[::4])
plt.plot([0, 4, 8, 12],test_sen_acc_num)
plt.legend(['kernel size = [1,3,5]', 'kernel size = [1,5,9]'])
plt.ylabel('sentence level acc')
plt.xlabel('num of epochs')
plt.title('different kernel size VS sentence level acc')
plt.savefig('./plots/sen_acc_kernel(100Vs200).png')
plt.close()
'''
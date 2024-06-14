
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
######################### load thư viện MNIST #################################
print("Load MNIST Database")
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()
x_train=np.reshape(x_train,(60000,784))/255.0                                   # scale giá trị điểm ảnh từ 0 tới 255
x_test= np.reshape(x_test,(10000,784))/255.0                                    # về từ 0 tới 1

y_train = np.matrix(np.eye(10)[y_train])                                        # tạo vecto one-hot 
y_test = np.matrix(np.eye(10)[y_test])                                          # ứng với nhãn gán (label)
print("----------------------------------") 
print(x_train.shape)
print(y_train.shape)
################ Định nghĩa các hàm sử dụng trong đoạn code ###################
def ReLu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def softmax(x):
    return np.divide(np.matrix(np.exp(x)),np.mat(np.sum(np.exp(x),axis=0)))

def Forwardpass(X,wh1,bh1,wh2,bh2,Wo,bo):                                       # hàm lan truyền thẳng
    x = X.transpose()
    # lớp ẩn thứ 1
    zh1 = wh1 @ x + bh1
    ah1 = ReLu(zh1)

    # lớp ẩn thứ 2
    zh2 = wh2 @ ah1 + bh2
    ah2 = sigmoid(zh2)
    
    # Ngõ ra
    z0 = wo @ ah2 + bo
    o = softmax(z0)
    return o

def AccTest(label,prediction):                                                  # hàm tính toán độ chính xác 
                                                                                # trên tập test
    label = label.transpose()
    OutMaxArg=np.argmax(prediction,axis=0)                                      # hàm argmax() sẽ trả về vị trí
    LabelMaxArg=np.argmax(label,axis=0)                                         # của phần tử lớn nhất trong mảng
    
    Accuracy=np.mean(OutMaxArg==LabelMaxArg)                                    
    return Accuracy


NumTrainSamples = x_train.shape[0]
NumTestSamples = x_test.shape[0]

####################### chọn số lượng nút trong mỗi lớp #######################

NumInputs = x_train.shape[1]
NumHiddenUnits1 = 50
NumHiddenUnits2 = 50
NumClasses = 10

####################### khởi tạo trọng số #####################################

# khởi tạo trọng số và bias tại lớp ẩn thứ 1
wh1=np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits1, NumInputs)))         # ma trận 10 x 720
bh1= np.random.uniform(0,0.5,(NumHiddenUnits1, 1))                              # ma trận 784 x 1

# khởi tạo trọng số và bias tại lớp ẩn thứ 2
wh2=np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits2, NumHiddenUnits1)))   # ma trận 50 x 50
bh2= np.random.uniform(0,0.5,(NumHiddenUnits2, 1))                              # ma trận 50 x 1

# khởi tạo trọng số và bias tại lớp ngõ ra
wo=np.random.uniform(-0.5,0.5,(NumClasses, NumHiddenUnits2))                    # ma trận 10 x 50
bo= np.random.uniform(0,0.5,(NumClasses, 1))                                    # ma trận 10 x 1

######## khởi tạo các mảng gradient tương ứng với từng lớp ####################
# gradient tại lớp ẩn thứ 1
dwh1= np.zeros((NumHiddenUnits1, NumInputs))                                    # ma trận 50 x 784
dbh1= np.zeros((NumHiddenUnits1, 1))                                            # ma trận 50 x 1

# gradient tại lớp ẩn thứ 2
dwh2= np.zeros((NumHiddenUnits2, NumHiddenUnits1))                              # ma trận 50 x 50
dbh2= np.zeros((NumHiddenUnits2, 1))                                            # ma trận 50 x 1

# gradient tại lớp ngõ ra
dwo= np.zeros((NumClasses, NumHiddenUnits2))                                    # ma trận 10 x 50
dbo= np.zeros((NumClasses, 1))                                                  # ma trận 10 x 1

############################ traning mô hình ##################################

from IPython.display import clear_output
loss = []
Acc = []
lr = 0.5                                                                        # tốc độ học
Epoch = 50
Batch_size = 200                                                                # chọn kích thước Batch
Stochastic_samples = np.arange(NumTrainSamples)                                 # tạo mảng có giá trị từ 0 tới 60000
for ep in range (Epoch):
  np.random.shuffle(Stochastic_samples)                                         # tráo đổi ngẫu nhiên các phần tử
                                                                                # trong mảng sau mỗi epoch
                                                                                
  for ite in range (0, NumTrainSamples, Batch_size):                            # ite sẽ chạy từ 0 tới 59999 
                                                                                # với step_size = 200 (batch_size)
                                                                                
    Batch_samples = Stochastic_samples[ite:ite+Batch_size]                      # lấy 200 mẫu sau mỗi iteration
    x = x_train[Batch_samples,:].transpose()
    y = y_train[Batch_samples,:].transpose()

#################### tính toán lan truyền thẳng ###############################
    
        # lớp ẩn thứ 1
    zh1 = wh1 @ x + bh1
    ah1 = ReLu(zh1)

        # lớp ẩn thứ 2
    zh2 = wh2 @ ah1 + bh2
    ah2 = sigmoid(zh2)
    
        # lớp ngõ ra
    z0 = wo @ ah2 + bo
    o = softmax(z0)

        # tính toán hàm mất mát bằng cross-entropy 
    loss.append(-np.sum(np.multiply(y,np.log10(o))))
    
##################### Tính toán lan truyền ngược ##############################

                #### lớp ngõ ra ####
    # sai số tại lớp ngõ ra     
    e0 = (o-y)            
    # tính toán gradient tại lớp ngõ ra
    dwo = e0 * np.transpose(ah2)
    db0 = np.mean(e0)
    
                #### lớp ẩn thứ 2 ####
    # sai số tại lớp ẩn thứ 2
    eh2 = np.multiply(np.multiply(np.matmul(np.transpose(wo), e0), ah2), (1-ah2))
    # tính toán gradient tại lớp ẩn thứ 2
    dwh2 = eh2 @ np.transpose(ah1)
    dbh2 = np.mean(eh2) 
            
                #### lớp ẩn thứ 1 ####
    # sai số tại lớp ẩn thứ 1
    eh1 = np.multiply(np.transpose(wh2) @ (eh2), np.heaviside(zh1, 0))          # hàm np.heaviside(x1, x2) sẽ trả về 
                                                                                ## 1 nếu x2 > x1
                                                                                ## 0 nếu x2 < x1
                                                                                ## x1 nếu x2 = x1 
                                                                                # tương tự như đạo hàm của hàm ReLU
                                                                                ## ReLU'(z) = 0 nếu z <=0
                                                                                ##          = 1 nếu z > 0
    # tính toán gradient tại lớp ẩn thứ 1
    dwh1 = eh1 @ np.transpose(x)
    dbh1 = np.mean(eh1)
    
########################## cập nhật trọng số ##################################
        # trọng số lớp ngõ ra
    wo = wo - lr * dwo / Batch_size
    bo = bo - lr * dbo
        # trọng số lớp ẩn thứ 1 
    wh1 = wh1 - lr * dwh1 / Batch_size
    bh1 = bh1 - lr * dbh1
        # trọng số lớp ẩn thứ 2
    wh2 = wh2 - lr * dwh2 / Batch_size
    bh2 = bh2 - lr * dbh2
    
    ###### độ chính xác của mô hình sau mỗi iteration ######
    prediction = Forwardpass(x_test,wh1,bh1,wh2,bh2,wo,bo)
    
    Acc.append(AccTest(y_test,prediction))                                      # lệnh sẽ trả về 
                                                                                # (số lượng phần tử True / kích thước tập test)
                                                                                # sau mỗi iteration
                                                                                # hay còn gọi là (độ chính xác)
                                                                                # của mô hình sau mỗi iteration
    clear_output(wait=True)
    plt.plot([i for i, _ in enumerate(Acc)],Acc,'o')
    plt.show()
    
  # in ra độ chính xác của mô hình sau mỗi epoch
  print('Epoch:', ep )
  print('Accuracy:',AccTest(y_test,prediction) )


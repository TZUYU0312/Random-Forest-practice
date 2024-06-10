using Random
using CSV
using DataFrames
using RDatasets
using DecisionTree
#file_path = "mtcars.csv"
data = dataset("datasets", "iris")
println(data)
Random.seed!(1234)
#function test_train_data(train_num,val_num,ts_num,data)
function test_train_data(train_num,ts_num,data)
    # 計算訓練集的大小，這裡我們假設使用 50% 的資料作為訓練集
    train_size = floor(Int,train_num*nrow(data))
    #val_size = floor(Int, val_num*nrow(data))
    #ts_size = floor(Int,ts_num*nrow(data))

    # 隨機抽樣索引以分割資料集
    indices = shuffle(1:nrow(data))
    train_indices = indices[1:train_size]
    #val_indices = indices[train_size+1:val_size+train_size]
    ts_indices = indices[train_size+1:end]
    return train_indices,ts_indices
end
# 計算訓練集的大小，這裡我們假設使用 50% 的資料作為訓練集
#train_set,val_set,ts_set = test_train_data(0.3,0.7,data)
train_set,ts_set = test_train_data(0.3,0.7,data)
train_data = data[train_set, :]
#val_data = data[val_set, :]
test_data = data[ts_set,:]
#println(train_data)
#println(val_data)
#println(test_data)

# 將資料轉換為 Arrays
features_train = Matrix(train_data[:, 1:4])
labels_train = Vector(train_data[:, 5])

features_test = Matrix(test_data[:, 1:4])
labels_test = Vector(test_data[:, 5])

# 訓練隨機森林模型
n_trees = 3  # 指定樹的數量
rf_model = build_forest(labels_train, features_train, n_trees)

# 對測試資料進行預測
predictions = apply_forest(rf_model, features_test)

global counts = 0
# 顯示預測結果
println(predictions)
# 計算準確率
accuracy = sum(predictions .== labels_test) / length(labels_test)
println("Accuracy: ", accuracy)

# 計算混淆矩陣
classes = unique(labels_test)
global new_label = String.(labels_test)
global new_pred = String.(predictions)
global new_label_test = []
global new_pred_data = []

for i in new_label
    push!(new_label_test,i)
end
for j in new_pred
    push!(new_pred_data,j)
end
# 將class和預測轉成整數
labels_int = map(label -> findfirst(classes .== label), labels_test)
predictions_int = map(pred -> findfirst(classes .== pred), predictions)


conf_matrix = confusmat(3,labels_int, predictions_int)



# 顯示混淆表格
println("Confusion Table:\n", conf_matrix)
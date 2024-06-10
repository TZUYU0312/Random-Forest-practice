using DataFrames, CSV, DecisionTree, ScikitLearn,Statistics,LinearAlgebra


function confusion_m(real_label, pred_label)
    global TP, FN, FP, TN
    if real_label == 1
        if pred_label == 1
            TP += 1
        else
            FN += 1
        end
    else
        if pred_label == 1
            FP += 1
        else
            TN += 1
        end
    end
end

function cal_result(rfmodel,file_path, real_label)
    
    test_file = CSV.read(file_path, DataFrame; header=false)
    rename!(test_file, Symbol.("col" .* string.(1:26)))

    cal_log_0 = 0.0
    cal_log_1 = 0.0
    

    for i in 1:nrow(test_file)
        prob = predict_proba(rfmodel, Matrix(test_file[i:i, :]))[1, :]
        cal_log_1 += log(prob[2] + eps())
        cal_log_0 += log(prob[1] + eps())
    end

    pred_label = cal_log_1 > cal_log_0 ? 1 : 0
    confusion_m(pred_label, real_label)
end


#驗證集命名規則nonvocal:0  vocal:1
train_0 = CSV.read("nonvocal/train/nonvocal_train.csv", DataFrame)
train_1 = CSV.read("vocal/train/vocal_train.csv", DataFrame)

global TP = 0
global FN = 0
global FP = 0
global TN = 0



# 添加欄位 (1 到 26)
column_names = Symbol.("col" .* string.(1:26))
rename!(train_1, column_names)
rename!(train_0, column_names)


insertcols!(train_0, :class => 0)
insertcols!(train_1, :class => 1)

train_df = vcat(train_0, train_1)

train_total = select(train_df, Not(:class))
features_label = train_df.class

# for i in 1:26
#     col_name = "col$i"
#     if col_name in names(train_total)
#         column_data = train_total[!, i]
#         std_dev = std(column_data)
#         mean_data = mean(column_data)
#         train_total[!, col_name] .= (column_data .- mean_data) / std_dev
#         println("The standard deviation of $col_name is: $std_dev")
#     else
#         println("Column $col_name not found in the DataFrame")
#     end
# end

# #計算covariance Matrix (6X6)
# Conv = (1/size(train_total,1)) * (Matrix(train_total)' * Matrix(train_total))


# # 計算covariance 的特徵值和特徵向量
# eigen_conv = eigen(Conv)

# # 取得特徵值與特真與特徵向量
# eigen_values = eigen_conv.values
# eigen_vector = eigen_conv.vectors

# reduce_eigen_values = eigen_values[1:23]

# #投影矩陣
# global Projection = eigen_vector[1,:]
# for i in 2:23
#     global Projection = hcat(Projection,eigen_vector[i,:])
# end

#計算出最後的轉換函數Y
#train_total = Matrix(train_total) * Projection

rf_model = DecisionTree.RandomForestClassifier(n_trees=80, max_depth=10)
ScikitLearn.fit!(rf_model, Matrix(train_total), features_label)


#算Accuracy
#vocal 測試集
for i in 0:399 
    file_path = "vocal/test/$(lpad(string(i), 4, '0')).csv"
    cal_result(rf_model, file_path, 1)
end

#nonvocal 測試集
for i in 0:199
    file_path = "nonvocal/test/$(lpad(string(i), 4, '0')).csv"
    cal_result(rf_model, file_path, 0)
end


accuracy = (TP + TN) / (TP + TN + FP + FN)

println("[TP, FN]: ","[$TP, $FN]")
println("[FP, TN]: ","[$FP, $TN]")
println("Accuracy: $accuracy")

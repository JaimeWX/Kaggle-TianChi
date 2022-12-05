import torch
from model import resnet34
import os
import pandas as pd


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #------------------ construct test dataset -----------------------------------------------------
    X_test = pd.read_csv("/wx/Models_WX/K-T/K_digit_recognizer/digit_recognizer/test.csv")
    arr_test = X_test.values
    #------------------ predicting: use ResNet34 ---------------------------------------------------
    # create model
    model = resnet34(num_classes=10).to(device)
    # load model weights
    weights_path = "/wx/Models_WX/K-T/K_digit_recognizer/resnet34_digit_recognizer.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # predicting
    model.eval()
    batch_size = 8
    # save results
    count = 1
    df  = pd.DataFrame(columns=["ImageId","Label"])
    with torch.no_grad():
        for ids in range(0,len(arr_test)//batch_size):
            img_list = []
            for np_img_flatten  in arr_test[ids * batch_size: (ids + 1) * batch_size]:
                np_img = np_img_flatten.reshape(28,28)
                tensor_img = torch.tensor(np_img)
                input_tensor_img = torch.unsqueeze(tensor_img,0) # torch.Size([28, 28, 1])
                input_tensor_img = input_tensor_img.type(torch.cuda.FloatTensor)
                img_list.append(input_tensor_img)
            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print(count)
                print("class: {}  prob: {:.3}".format(str(cla.numpy()),pro.numpy()))
                df.loc[len(df.index)] = [count,cla.numpy()]
                count += 1
    df.to_csv("K_digit_recognizer.csv", index=False)
    print("done.")
                
if __name__ == '__main__':
    main()

    # df = pd.read_csv("/wx/Models_WX/K-T/K_digit_recognizer/K_digit_recognizer.csv")
    # df = df.loc[ : , ~df.columns.str.contains('Unnamed')]
    # df.to_csv("/wx/Models_WX/K-T/K_digit_recognizer/K_digit_recognizer.csv", index=False)
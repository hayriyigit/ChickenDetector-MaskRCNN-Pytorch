import torch
import cv2
import config
import numpy as np


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(state, filename="mask_rcnn.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def predict_single_frame(frame, model):
    images = cv2.resize(frame, config.IMAGE_SIZE, cv2.INTER_LINEAR)
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images = images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(config.DEVICE) for image in images)

    with torch.no_grad():
        pred = model(images)

    im = images[0].swapaxes(0, 2).swapaxes(
        0, 1).detach().cpu().numpy().astype(np.uint8)
    im2 = np.zeros_like(im).astype(np.uint8)
    for i in range(len(pred[0]['masks'])):
        msk = pred[0]['masks'][i, 0].detach().cpu().numpy()
        scr = pred[0]['scores'][i].detach().cpu().numpy()
        box = pred[0]['boxes'][i].detach().cpu().numpy()

        if scr > 0.87:
            cv2.rectangle(im, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (0, 0, 1), 2)
            cv2.putText(im, "{0:.2f}%".format(scr*100), (int(box[0]+5), int(box[1])+15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 1), 2, cv2.LINE_AA)
            im2[:, :, 0][msk > 0.87] = np.random.randint(0,255)
            im2[:, :, 1][msk > 0.87] = np.random.randint(0,255)
            im2[:, :, 2][msk > 0.87] = np.random.randint(0,255)

    return cv2.addWeighted(im, 0.8, im2, 0.2,0)


def predict_video(input, output, model):
    cap = cv2.VideoCapture(input)
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(
        'M', 'P', '4', 'V'), 60, (640, 640))
    model.train(False)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            result_frame = predict_single_frame(frame, model)
            out.write(result_frame)
        else:
            break

    cap.release()
    out.release()

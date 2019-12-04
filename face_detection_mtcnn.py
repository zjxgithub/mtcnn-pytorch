from src import detect_faces, show_bboxes
from PIL import Image
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
img_path = '/net/deepfake-defense/datasets/CelebA/img/img_celeba/'
pert_path = '/net/deepfake-defense/datasets/CelebA/img/MTCNN_ifgsm/'
result_f = open('/home/zhujunxiao/protect_face_identity/face_image_protection/MTCNN/mtcnn-pytorch/results/MTCNN_detection_result_celeba.csv', 'w')
result_f.write('image, detected bounding box, detected boundingbox(after perturbation)\n')
for img_index in range(1, 2001):
    img_filename = format(img_index, '06d') + '.jpg'
    print(img_filename)
    img = Image.open(img_path +  img_filename)
    img = img.resize((224, 224))
    bounding_boxes, landmarks = detect_faces(img)
    bounding_box_str = []
    for box in bounding_boxes:
        bounding_box_str.append(' '.join([str(x) for x in box]))
    # for i in range(len(bounding_boxes)):
    #     result_f.write(img_filename + ',' + ' '.join([str(x) for x in bounding_boxes[i]]))
    #     result_f.write(',' + ' '.join([str(x) for x in landmarks[i]]) + '\n')
    pert_img =  pert_path + img_filename
    img = Image.open(pert_img)
    img = img.resize((224, 224))
    pert_bounding_boxes, _ = detect_faces(img)
    pert_bounding_box_str = []
    for box in pert_bounding_boxes:
        pert_bounding_box_str.append(' '.join([str(x) for x in box]))
    result_f.write(img_filename  + ',')
    result_f.write(';'.join([x for x in bounding_box_str]) + ',')
    result_f.write(';'.join([x for x in pert_bounding_box_str]) + '\n')

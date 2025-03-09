#include <iostream>
#include <vector>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/image_io.h>
#include <dlib/dnn.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace dlib;

// ------------------- 网络定义 -------------------
// 以下网络定义基于 dlib 官方示例，用于人脸识别模型 dlib_face_recognition_resnet_model_v1.dat
// ----------------------------------------------------------------------------------------
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    alevel0<
    alevel1<
    alevel2<
    alevel3<
    alevel4<
    max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
    input_rgb_image_sized<150>
    >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------


// ------------------- 网络定义结束 -------------------

int main() {
    try {
        // 1. 加载两张输入图像
        cv::Mat img1 = cv::imread("D:\\OpenCV_images\\b111.jpg");
        cv::Mat img2 = cv::imread("D:\\OpenCV_images\\b222.jpg");
        if (img1.empty() || img2.empty()) {
            cout << "Could not read one of the image files!" << endl;
            return -1;
        }

        // 2. 初始化面部检测器和形状预测器
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize("D:\\OpenCV_images\\shape_predictor_68_face_landmarks.dat") >> sp;

        // 3. 将 OpenCV 图像转换为 dlib 格式（注意：cv::Mat 的 BGR 顺序会自动转换为 RGB）
        cv_image<rgb_pixel> dlibImg1(img1);
        cv_image<rgb_pixel> dlibImg2(img2);

        // 4. 检测人脸（这里假设每张图片仅含一个人脸，选择检测到的第一个人脸）
        std::vector<rectangle> faces1 = detector(dlibImg1);
        std::vector<rectangle> faces2 = detector(dlibImg2);
        if (faces1.empty() || faces2.empty()) {
            cout << "No faces found in one of the images!" << endl;
            return -1;
        }

        rectangle face1 = faces1[0];
        rectangle face2 = faces2[0];

        // 5. 提取人脸特征点，并提取对齐后的 face chip
        full_object_detection shape1 = sp(dlibImg1, face1);
        full_object_detection shape2 = sp(dlibImg2, face2);

        matrix<rgb_pixel> chip1, chip2;
        extract_image_chip(dlibImg1, get_face_chip_details(shape1, 150, 0.25), chip1);
        extract_image_chip(dlibImg2, get_face_chip_details(shape2, 150, 0.25), chip2);

        // 6. 初始化人脸识别模型（ResNet模型），使用 anet_type
        anet_type net;
        deserialize("D:\\OpenCV_images\\dlib_face_recognition_resnet_model_v1.dat") >> net;

        // 7. 计算两张人脸的特征向量
        matrix<float, 0, 1> descriptor1 = net(chip1);
        matrix<float, 0, 1> descriptor2 = net(chip2);

        // 8. 比较特征向量，计算欧氏距离
        double distance = length(descriptor1 - descriptor2);
        cout << "Face descriptor distance: " << distance << endl;
        if (distance < 0.6)
            cout << "These images are likely of the same person!" << endl;
        else
            cout << "These images are probably not the same person." << endl;

    } catch (std::exception& e) {
        cout << "Exception thrown: " << e.what() << endl;
        return -1;
    }

    return 0;
}

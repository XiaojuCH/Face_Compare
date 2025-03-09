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

// ------------------- ���綨�� -------------------
// �������綨����� dlib �ٷ�ʾ������������ʶ��ģ�� dlib_face_recognition_resnet_model_v1.dat
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


// ------------------- ���綨����� -------------------

int main() {
    try {
        // 1. ������������ͼ��
        cv::Mat img1 = cv::imread("D:\\OpenCV_images\\b111.jpg");
        cv::Mat img2 = cv::imread("D:\\OpenCV_images\\b222.jpg");
        if (img1.empty() || img2.empty()) {
            cout << "Could not read one of the image files!" << endl;
            return -1;
        }

        // 2. ��ʼ���沿���������״Ԥ����
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize("D:\\OpenCV_images\\shape_predictor_68_face_landmarks.dat") >> sp;

        // 3. �� OpenCV ͼ��ת��Ϊ dlib ��ʽ��ע�⣺cv::Mat �� BGR ˳����Զ�ת��Ϊ RGB��
        cv_image<rgb_pixel> dlibImg1(img1);
        cv_image<rgb_pixel> dlibImg2(img2);

        // 4. ����������������ÿ��ͼƬ����һ��������ѡ���⵽�ĵ�һ��������
        std::vector<rectangle> faces1 = detector(dlibImg1);
        std::vector<rectangle> faces2 = detector(dlibImg2);
        if (faces1.empty() || faces2.empty()) {
            cout << "No faces found in one of the images!" << endl;
            return -1;
        }

        rectangle face1 = faces1[0];
        rectangle face2 = faces2[0];

        // 5. ��ȡ���������㣬����ȡ������ face chip
        full_object_detection shape1 = sp(dlibImg1, face1);
        full_object_detection shape2 = sp(dlibImg2, face2);

        matrix<rgb_pixel> chip1, chip2;
        extract_image_chip(dlibImg1, get_face_chip_details(shape1, 150, 0.25), chip1);
        extract_image_chip(dlibImg2, get_face_chip_details(shape2, 150, 0.25), chip2);

        // 6. ��ʼ������ʶ��ģ�ͣ�ResNetģ�ͣ���ʹ�� anet_type
        anet_type net;
        deserialize("D:\\OpenCV_images\\dlib_face_recognition_resnet_model_v1.dat") >> net;

        // 7. ����������������������
        matrix<float, 0, 1> descriptor1 = net(chip1);
        matrix<float, 0, 1> descriptor2 = net(chip2);

        // 8. �Ƚ���������������ŷ�Ͼ���
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

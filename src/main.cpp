#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main() {
    // Load first image
    std::string img_path = "C:\\Projects\\VIO\\data\\MH_01_easy\\MH_01_easy\\mav0\\cam0\\data\\1403636579763555584.png";
    
    cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    
    if (img.empty()) {
        std::cerr << "Failed to load image! Check path." << std::endl;
        return -1;
    }
    
    std::cout << "✓ Image loaded: " << img.cols << "x" << img.rows << std::endl;
    
    // Detect ORB features
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> keypoints;
    
    orb->detect(img, keypoints);
    
    std::cout << "✓ Detected " << keypoints.size() << " features" << std::endl;
    
    // Draw and display
    cv::Mat img_with_keypoints;
    cv::drawKeypoints(img, keypoints, img_with_keypoints, cv::Scalar(0, 255, 0));
    
    cv::imshow("Features", img_with_keypoints);
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    
    return 0;
}
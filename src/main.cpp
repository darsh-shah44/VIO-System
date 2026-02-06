#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main() {
    std::string img1_path = "C:\\Projects\\VIO\\data\\MH_01_easy\\MH_01_easy\\mav0\\cam0\\data\\1403636579763555584.png";
    std::string img2_path = "C:\\Projects\\VIO\\data\\MH_01_easy\\MH_01_easy\\mav0\\cam0\\data\\1403636580163555584.png";

    cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);
    
    if (img1.empty() || img2.empty()) {
        std::cerr << "Failed to load images! Check path." << std::endl;
        return -1;
    }
    
    std::cout << "✓ Images Loaded" << std::endl;
    
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    orb->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);
    
    std::cout << "✓ Detected " << keypoints1.size() << " features in image 1 and " << keypoints2.size() << " in image 2" << std::endl;
    
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;

    matcher.match(descriptors1, descriptors2, matches);

    std::cout << "✓ Found " << matches.size() << " matches" << std::endl;

    std::vector<cv::Point2f> points1, points2;
    for(const auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    std::cout << "✓ Extracted " << points1.size() << " point correspondences" << std::endl;

    double focal_length = 458.654;
    cv::Point2d principal_point(367.215, 248.375);

    cv::Mat E = cv::findEssentialMat(points1, points2, focal_length, principal_point, cv::RANSAC);
    
    cv::Mat R, t;
    cv::recoverPose(E, points1, points2, R, t, focal_length, principal_point);
    
    std::cout << "\n=== Camera Motion ===" << std::endl;
    std::cout << "Rotation matrix R:\n" << R << std::endl;
    std::cout << "\nTranslation vector t:\n" << t << std::endl;

    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    
    cv::imshow("Matches", img_matches);
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);

    return 0;
}
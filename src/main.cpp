#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main() {
    // Load in two images from the dataset
    std::string img1_path = "C:\\Projects\\VIO\\data\\MH_01_easy\\MH_01_easy\\mav0\\cam0\\data\\1403636579763555584.png";
    std::string img2_path = "C:\\Projects\\VIO\\data\\MH_01_easy\\MH_01_easy\\mav0\\cam0\\data\\1403636580163555584.png";

    cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);
    
    if (img1.empty() || img2.empty()) {
        std::cerr << "Failed to load images! Check path." << std::endl;
        return -1;
    }
    
    std::cout << "✓ Images Loaded" << std::endl;

    // Detect ORB features - detects corners and edges, and computes binary descriptors
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    // Finds keypoints and computes descriptors in one step
    orb->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);
    
    std::cout << "✓ Detected " << keypoints1.size() << " features in image 1 and " << keypoints2.size() << " in image 2" << std::endl;
    
    // Brute force matcher with Hamming distance for binary descriptors
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;

    // Find the closest match for each feature in img 1 to features in img 2
    matcher.match(descriptors1, descriptors2, matches);

    std::cout << "✓ Found " << matches.size() << " matches" << std::endl;

    //Extract the actual point coordinates from the keypoints for the matched features
    std::vector<cv::Point2f> points1, points2;
    for(const auto& match : matches) {
        // queryIdx is the index of the descriptor in descriptors1, trainIdx is the index in descriptors2
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    std::cout << "✓ Extracted " << points1.size() << " point correspondences" << std::endl;

    //Camera intrinsic parameters (from EuRoC calibration file)
    double focal_length = 458.654; //focal length in pixels
    cv::Point2d principal_point(367.215, 248.375); //optical center of the camera

    // Compute the essential matrix and use RANSAC to filter out outliers
    cv::Mat E = cv::findEssentialMat(points1, points2, focal_length, principal_point, cv::RANSAC);
    
    // Get rotation and translation from the essential matrix
    cv::Mat R, t;
    cv::recoverPose(E, points1, points2, R, t, focal_length, principal_point);
    

    // Display the results
    std::cout << "\n=== Camera Motion ===" << std::endl;
    std::cout << "Rotation matrix R:\n" << R << std::endl;
    std::cout << "\nTranslation vector t:\n" << t << std::endl;

    // Draw matches for visualization
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    
    cv::imshow("Matches", img_matches);
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);

    return 0;
}
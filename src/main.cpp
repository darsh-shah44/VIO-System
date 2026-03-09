#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main() {
    // Load in two images from the dataset
    std::vector<std::string> image_paths;
    cv::glob("C:\\Projects\\VIO\\data\\MH_01_easy\\MH_01_easy\\mav0\\cam0\\data\\*.png", image_paths, false);
    
    cv::Mat R_total = cv::Mat::eye(3, 3, CV_64F); // Initialize total rotation as identity
    cv::Mat t_total = cv::Mat::zeros(3, 1, CV_64F); // Initialize total translation as zero
    
    // Detect ORB features - detects corners and edges, and computes binary descriptors
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);

    //Camera intrinsic parameters (from EuRoC calibration file)
        double focal_length = 458.654; //focal length in pixels
        cv::Point2d principal_point(367.215, 248.375); //optical center of the camera

    for (int i = 0; i < image_paths.size() - 1; i++){  // Processing pairs of consecutive images
        cv::Mat img1 = cv::imread(image_paths[i], cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread(image_paths[i + 1], cv::IMREAD_GRAYSCALE);

        if (img1.empty() || img2.empty()) {
            std::cerr << "Error: Could not load images at " << image_paths[i] << " and " << image_paths[i + 1] << std::endl;
            return -1;
        }

        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
    
        // Finds keypoints and computes descriptors in one step
        orb->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
        orb->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);
    
        
        // Brute force matcher with Hamming distance for binary descriptors
        cv::BFMatcher matcher(cv::NORM_HAMMING);

        std::vector<std::vector<cv::DMatch>> knn_matches;
        // Find the closest match for each feature in img 1 to features in img 2
        matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

        std::vector<cv::DMatch> good_matches;
        for (const auto& pair : knn_matches) {
            if (pair[0].distance < 0.75f * pair[1].distance) {
                good_matches.push_back(pair[0]);
            }
        }

        //Extract the actual point coordinates from the keypoints for the matched features
        std::vector<cv::Point2f> points1, points2;
        for(const auto& match : good_matches) {
            // queryIdx is the index of the descriptor in descriptors1, trainIdx is the index in descriptors2
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }

        // Compute the essential matrix and use RANSAC to filter out outliers
        cv::Mat E = cv::findEssentialMat(points1, points2, focal_length, principal_point, cv::RANSAC);
    
        // Get rotation and translation from the essential matrix
        cv::Mat R, t;
        cv::recoverPose(E, points1, points2, R, t, focal_length, principal_point);
        R_total = R * R_total; // Update total rotation
        t_total = t_total + R_total * t; // Update total translation taking into account the current rotation
    

        // Display the results
        std::cout << "\n=== Camera Motion ===" << std::endl;
        std::cout << "Cumulative position (frame: " << i + 1 << "):" << std::endl;
        std::cout << t_total << std::endl;
    }

    return 0;
}
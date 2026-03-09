#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <fstream>

int main() {
    // Load in two images from the dataset
    std::vector<std::string> image_paths;
    std::string data_path = "C:\\Projects\\VIO\\data\\MH_01_easy\\MH_01_easy\\mav0\\cam0\\data\\*.png";
    cv::glob(data_path, image_paths, false);
    
    if (image_paths.empty()) {
        std::cerr << "Error: No images found. Please ensure the image files are extracted to:" << std::endl;
        std::cerr << "C:\\Projects\\VIO\\data\\MH_01_easy\\MH_01_easy\\mav0\\cam0\\data\\" << std::endl;
        std::cerr << std::endl << "Searched pattern: " << data_path << std::endl;
        return -1;
    }
    
    std::ofstream trajectoryFile;
    trajectoryFile.open("trajectory.csv");
    trajectoryFile << "frame,x,y,z" << std::endl; // Write CSV header

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
            if (pair.size() < 2) continue;
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

        // Need at least 8 points for essential matrix
        if (points1.size() < 8 || points2.size() < 8) {
            std::cerr << "Warning: Frame " << i << " has only " << points1.size() << " matches. Skipping." << std::endl;
            continue;
        }

        // Compute the essential matrix and use RANSAC to filter out outliers
        cv::Mat E = cv::findEssentialMat(points1, points2, focal_length, principal_point, cv::RANSAC);
    
        // Check if essential matrix is valid
        if (E.empty()) {
            std::cerr << "Warning: Could not compute essential matrix for frame " << i << ". Skipping." << std::endl;
            continue;
        }
        
        // Get rotation and translation from the essential matrix
        cv::Mat R, t;
        int success = cv::recoverPose(E, points1, points2, R, t, focal_length, principal_point);
        
        if (success == 0) {
            std::cerr << "Warning: Could not recover pose for frame " << i << ". Skipping." << std::endl;
            continue;   
        }
        
        
        R_total = R * R_total; // Update total rotation
        t_total = t_total + R_total * t; // Update total translation taking into account the current rotation
    

        trajectoryFile << i + 1 << "," << t_total.at<double>(0) << "," << t_total.at<double>(1) << "," << t_total.at<double>(2) << std::endl; // Write current frame and position to CSV

    }
    
    trajectoryFile.close(); // Explicitly close the file

    return 0;
}
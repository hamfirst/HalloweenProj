///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltru�aitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltru�aitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltru�aitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltru�aitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////
// FaceTrackingVid.cpp : Defines the entry point for the console application for tracking faces in videos.

// Libraries for landmark detection (includes CLNF and CLM modules)
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"

#include <SequenceCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>

#include <opencv2/imgproc.hpp>

#ifdef _MSC_VER
#include <winsock2.h>
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort(const std::string & error)
{
	std::cout << error << std::endl;
	abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

cv::Point2f RotateLine(const cv::Point2f & line)
{
	return cv::Point2f(line.y, -line.x);
}

void ComputeAverageDir(const std::vector<cv::Point2f> & points, cv::Point2f & out_dir)
{
	out_dir = cv::Point2f(0, 0);

	for(size_t i = 0; i < points.size(); ++i)
	{
		auto & a = points[i];
		for(size_t j = i + 1; j < points.size(); ++j)
		{
			auto & b = points[j];
			auto offset = a - b;

			auto mag_sqrd = offset.x * offset.x + offset.y * offset.y;
			if(mag_sqrd == 0)
			{
				continue;
			}

			auto dir = offset;
			if(dir.x * out_dir.x + dir.y + out_dir.y < 0)
			{
				dir.x *= -1;
				dir.y *= -1;
			}

			out_dir += dir;
		}
	}

	auto total_mag_sqrd = out_dir.x * out_dir.x + out_dir.y * out_dir.y;
	if(total_mag_sqrd == 0)
	{
		return;
	}

	auto total_mag = sqrtf(total_mag_sqrd);

	out_dir.x /= total_mag;
	out_dir.y /= total_mag;
}

void ComputeAveragePos(const std::vector<cv::Point2f> & points, const cv::Point2f & dir, cv::Point2f & out_pos)
{
	out_pos = cv::Point2f(0, 0);

	auto norm_dir = RotateLine(dir);
	float d_total = 0;
	float num = 0;

	for(auto & elem : points)
	{
		d_total += norm_dir.x * elem.x + norm_dir.y * elem.y;
		num += 1;
	}

	if(num > 0)
	{
		d_total /= num;
		out_pos.x = norm_dir.x * d_total;
		out_pos.y = norm_dir.y * d_total;
	}
}

void ConstructLine(const std::vector<cv::Point2f> & points, cv::Point2f & out_pos, cv::Point2f & out_dir)
{
	ComputeAverageDir(points, out_dir);
	ComputeAveragePos(points, out_dir, out_pos);
}

void ConstructLine(const cv::Mat1f & landmarks, const std::vector<int> indices, cv::Point2f & out_pos, cv::Point2f & out_dir)
{
	int n = landmarks.rows / 2;
	std::vector<cv::Point2f> points;
	for(auto & i : indices)
	{
		cv::Point2f featurePoint(landmarks.at<float>(i), landmarks.at<float>(i + n));
		points.push_back(featurePoint);
	}

	ConstructLine(points, out_pos, out_dir);
}

cv::Point2f AlignLines(const std::vector<cv::Point2f> & lines)
{
	cv::Point2f dir(0, 0);
	
	for(auto & elem : lines)
	{
		auto dp = elem.dot(dir);
		if(dp < 0)
		{
			dir -= elem;
		}
		else
		{
			dir += elem;
		}
	}

	auto mag = sqrtf(dir.x * dir.x + dir.y * dir.y);
	if(mag > 0)
	{
		dir.x /= mag;
		dir.y /= mag;
	}

	return dir;
}

cv::Point2f MatchDirection(const cv::Point2f & dir, const cv::Point2f & a, const cv::Point2f & b)
{
	auto offset = a - b;
	if(dir.dot(offset) < 0)
	{
		return -dir;
	}

	return dir;
}

void DrawLandmarkLine(Utilities::Visualizer & visualizer, const cv::Mat1f & landmarks, const std::vector<int> indices)
{
	cv::Point2f out_pos = {};
	cv::Point2f out_dir = {};

	ConstructLine(landmarks, indices, out_pos, out_dir);
	visualizer.DrawLine(out_pos, out_dir);
}

struct PosDir
{
	PosDir() = default;
	PosDir(const cv::Point2f & pos, const cv::Point2f & dir)
	{
		m_Pos = pos;
		m_Dir = dir;
		m_DirD = m_Dir.dot(m_Pos);
		m_RotatedD = RotateLine(dir).dot(m_Pos);
	}

	PosDir(const PosDir & rhs) = default;
	PosDir(PosDir && rhs) noexcept = default;

	PosDir & operator =(const PosDir & rhs) = default;
	PosDir & operator =(PosDir && rhs) noexcept = default;

	cv::Point2f m_Pos;
	cv::Point2f m_Dir;

	float m_DirD;
	float m_RotatedD;
};

struct ImportandLandmarks
{
	PosDir m_LeftSide;
	PosDir m_RightSide;
	PosDir m_TopSide;
	PosDir m_BottomSide;
	PosDir m_CenterLine;
	PosDir m_NoseLine;
	PosDir m_BridgeLine;
	PosDir m_MouthLine;
	PosDir m_MouthExtentLine;

	cv::Point2f m_OverallDir;
	cv::Point2f m_RotatedDir;

	float m_FaceWidth;
	float m_ChinToNose;
	float m_BridgeToNose;
	float m_MouthExtent;
};

float CalculateLineDistance(const PosDir & a, const PosDir & b)
{
	return fabs(a.m_RotatedD - b.m_RotatedD);
}

void CalculateImportantLandmarkInfo(const cv::Mat1f & landmarks, ImportandLandmarks & out_landmarks)
{
	struct BoundInfo
	{
		std::vector<int> m_Points;
		bool m_Rotate;
	};

	std::vector<BoundInfo> bound_info = 
	{
		BoundInfo{ {0, 1, 2}, false },
		BoundInfo{ {14, 15, 16}, false },
		BoundInfo{ {7, 8, 9}, true },
		BoundInfo{ {24, 19}, true },
		BoundInfo{ {30, 8}, false },
		BoundInfo{ {31, 32, 33, 34, 35}, true },
		BoundInfo{ {28}, true }
	};

	std::vector<cv::Point2f> dirs;
	for(auto & elem : bound_info)
	{
		int n = landmarks.rows / 2;
		std::vector<cv::Point2f> points;
		for(auto & i : elem.m_Points)
		{
			cv::Point2f featurePoint(landmarks.at<float>(i), landmarks.at<float>(i + n));
			points.push_back(featurePoint);
		}

		cv::Point2f dir;
		ComputeAverageDir(points, dir);

		if(elem.m_Rotate)
		{
			dir = RotateLine(dir);
		}

		dirs.push_back(dir);
	}


	int n = landmarks.rows / 2;

	auto bridge_pos = cv::Point2f(landmarks.at<float>(28), landmarks.at<float>(28 + n));
	auto nose_pos = cv::Point2f(landmarks.at<float>(30), landmarks.at<float>(30 + n));

	auto overall_dir = AlignLines(dirs);
	overall_dir = MatchDirection(overall_dir, bridge_pos, nose_pos);
	auto rotated_dir = RotateLine(overall_dir);

	std::vector<PosDir> out_posdir;

	for(auto & elem : bound_info)
	{
		cv::Point2f pos;

		std::vector<cv::Point2f> points;
		for(auto & i : elem.m_Points)
		{
			cv::Point2f featurePoint(landmarks.at<float>(i), landmarks.at<float>(i + n));
			points.push_back(featurePoint);
		}

		auto dir = elem.m_Rotate ? rotated_dir : overall_dir;

		ComputeAveragePos(points, dir, pos);
		out_posdir.emplace_back(PosDir{pos, dir});
	}

	out_landmarks.m_RightSide = out_posdir[0];
	out_landmarks.m_LeftSide = out_posdir[1];
	out_landmarks.m_BottomSide = out_posdir[2];
	out_landmarks.m_TopSide = out_posdir[3];
	out_landmarks.m_CenterLine = out_posdir[4];
	out_landmarks.m_NoseLine  = out_posdir[5];
	out_landmarks.m_BridgeLine = out_posdir[6];
	out_landmarks.m_FaceWidth = CalculateLineDistance(out_landmarks.m_RightSide, out_landmarks.m_LeftSide);
	out_landmarks.m_ChinToNose = CalculateLineDistance(out_landmarks.m_NoseLine, out_landmarks.m_BottomSide);
	out_landmarks.m_BridgeToNose = CalculateLineDistance(out_landmarks.m_BridgeLine, out_landmarks.m_NoseLine);

	out_landmarks.m_OverallDir = overall_dir;
	out_landmarks.m_RotatedDir = rotated_dir;

	auto mouth_pos = nose_pos - overall_dir * (out_landmarks.m_BridgeToNose);
	auto mouth_extent_pos = nose_pos - overall_dir * (out_landmarks.m_BridgeToNose * 2);
	out_landmarks.m_MouthLine = PosDir{mouth_pos, rotated_dir};
	out_landmarks.m_MouthExtentLine = PosDir{mouth_extent_pos, rotated_dir};
	out_landmarks.m_MouthExtent = CalculateLineDistance(out_landmarks.m_NoseLine, out_landmarks.m_MouthExtentLine);
}

cv::Point2f NormalizeMouthPos(const cv::Mat1f & landmarks, const ImportandLandmarks & landmark_info, int pos_index)
{
	int n = landmarks.rows / 2;
	auto pos = cv::Point2f(landmarks.at<float>(pos_index), landmarks.at<float>(pos_index + n));

	auto x_d = landmark_info.m_RotatedDir.dot(pos) - landmark_info.m_LeftSide.m_RotatedD;
	auto y_d = RotateLine(landmark_info.m_RotatedDir).dot(pos) - landmark_info.m_NoseLine.m_RotatedD;

	//printf("%f, %f, %f, %f\n", landmark_info.m_NoseLine.m_RotatedD, landmark_info.m_MouthExtentLine.m_RotatedD, y_d, landmark_info.m_MouthExtent);

	x_d /= landmark_info.m_FaceWidth;
	y_d /= landmark_info.m_MouthExtent;

	x_d = 1.0f - x_d;

	return cv::Point2f(x_d, y_d);
}

int main(int argc, char **argv)
{
#ifdef _MSC_VER
	auto iResult = WSAStartup(MAKEWORD(2,2), &wsaData);
	if (iResult != 0) 
	{
		printf("WSAStartup failed: %d\n", iResult);
		return 1;
	}
#endif
	int sock;
    if ((sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
	{
        printf("socket() failed\n");
		return 1;
	}

	int broadcastPermission = 1;
    if (setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcastPermission, sizeof(broadcastPermission)) < 0)
	{
        printf("setsockopt() failed\n");
		return 1;
	}

	sockaddr_in broadcastAddr;
	memset(&broadcastAddr, 0, sizeof(broadcastAddr));
    broadcastAddr.sin_family = AF_INET;
    broadcastAddr.sin_addr.s_addr = inet_addr("255.255.255.255");
    broadcastAddr.sin_port = htons(51223);

	vector<string> arguments = get_arguments(argc, argv);

	// no arguments: output usage
	if (arguments.size() == 1)
	{
		cout << "For command line arguments see:" << endl;
		cout << " https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments";
		return 0;
	}

	LandmarkDetector::FaceModelParameters det_parameters(arguments);

	// The modules that are being used for tracking
	LandmarkDetector::CLNF face_model(det_parameters.model_location);
	if (!face_model.loaded_successfully)
	{
		cout << "ERROR: Could not load the landmark detector" << endl;
		return 1;
	}

	if (!face_model.eye_model)
	{
		cout << "WARNING: no eye model found" << endl;
	}

	// Open a sequence
	Utilities::SequenceCapture sequence_reader;

	// A utility for visualizing the results (show just the tracks)
	Utilities::Visualizer visualizer(true, false, false, false);

	// Tracking FPS for visualization
	Utilities::FpsTracker fps_tracker;
	fps_tracker.AddFrame();

	int sequence_number = 0;

	while (true) // this is not a for loop as we might also be reading from a webcam
	{

		// The sequence reader chooses what to open based on command line arguments provided
		if (!sequence_reader.Open(arguments))
			break;

		INFO_STREAM("Device or file opened");

		cv::Mat rgb_image = sequence_reader.GetNextFrame();

		INFO_STREAM("Starting tracking");
		while (!rgb_image.empty()) // this is not a for loop as we might also be reading from a webcam
		{

			// Reading the images
			cv::Mat_<uchar> grayscale_image = sequence_reader.GetGrayFrame();

			// The actual facial landmark detection / tracking
			bool detection_success = LandmarkDetector::DetectLandmarksInVideo(rgb_image, face_model, det_parameters, grayscale_image);

			// Gaze tracking, absolute gaze direction
			cv::Point3f gazeDirection0(0, 0, -1);
			cv::Point3f gazeDirection1(0, 0, -1);

			// If tracking succeeded and we have an eye model, estimate gaze
			if (detection_success && face_model.eye_model)
			{
				GazeAnalysis::EstimateGaze(face_model, gazeDirection0, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true);
				GazeAnalysis::EstimateGaze(face_model, gazeDirection1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, false);
			}

			// Work out the pose of the head from the tracked model
			cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);

			// Keeping track of FPS
			fps_tracker.AddFrame();

			// Displaying the tracking visualizations
			visualizer.SetImage(rgb_image, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);

			ImportandLandmarks important_landmarks;
			CalculateImportantLandmarkInfo(face_model.detected_landmarks, important_landmarks);

			visualizer.DrawLine(important_landmarks.m_RightSide.m_Pos, important_landmarks.m_RightSide.m_Dir);
			visualizer.DrawLine(important_landmarks.m_LeftSide.m_Pos, important_landmarks.m_LeftSide.m_Dir);
			visualizer.DrawLine(important_landmarks.m_NoseLine.m_Pos, important_landmarks.m_NoseLine.m_Dir);
			visualizer.DrawLine(important_landmarks.m_MouthLine.m_Pos, important_landmarks.m_MouthLine.m_Dir);
			visualizer.DrawLine(important_landmarks.m_MouthExtentLine.m_Pos, important_landmarks.m_MouthExtentLine.m_Dir);

			cv::Point2f mouth_positions[20];
			for(int index = 0, point = 48; index < 20; ++index, ++point)
			{
				mouth_positions[index] = NormalizeMouthPos(face_model.detected_landmarks, important_landmarks, point);
			}

			cv::Point2f prev_point = mouth_positions[0];
			cv::Point2f cur_point = mouth_positions[1];

			for(int index = 2; index < 20; ++index)
			{
				visualizer.DrawLineSeg(prev_point * 300.0f, cur_point * 300.0f);
				prev_point = cur_point;
				cur_point = mouth_positions[index];
			}

			visualizer.DrawLineSeg(cv::Point2f(0, 300), cv::Point2f(300, 300));
			visualizer.DrawLineSeg(cv::Point2f(300, 0), cv::Point2f(300, 300));

			if(sendto(sock, mouth_positions, sizeof(mouth_positions), 0, (sockaddr *)&broadcastAddr, sizeof(broadcastAddr)) < 0)
			{
				printf("Failed to send\n");
				return -1; 
			};

			visualizer.SetObservationLandmarks(face_model.detected_landmarks, face_model.detection_certainty, face_model.GetVisibilities());
			//visualizer.SetObservationPose(pose_estimate, face_model.detection_certainty);
			//visualizer.SetObservationGaze(gazeDirection0, gazeDirection1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy), face_model.detection_certainty);
			visualizer.SetFps(fps_tracker.GetFPS());

			// detect key presses (due to pecularities of OpenCV, you can get it when displaying images)
			char character_press = visualizer.ShowObservation();


			// restart the tracker
			if (character_press == 'r')
			{
				face_model.Reset();
			}
			// quit the application
			else if (character_press == 'q')
			{
				return(0);
			}

			// Grabbing the next frame in the sequence
			rgb_image = sequence_reader.GetNextFrame();

		}

		// Reset the model, for the next video
		face_model.Reset();
		sequence_reader.Close();

		sequence_number++;

	}
	return 0;
}


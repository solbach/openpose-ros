//#include<iostream>
#define USE_CAFFE

#include <gflags/gflags.h>
// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

#include <glog/logging.h>

#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

#include "ros/ros.h"
#include <std_srvs/Empty.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <openpose_ros/Person.h>

DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_path,               "examples/media/COCO_val2014_000000000192.jpg",     "Process the desired image.");
// OpenPose
DEFINE_string(model_pose,               "COCO",         "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                                        "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution,           "-1x368",       "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
                                                        " decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
                                                        " closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
                                                        " any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
                                                        " input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
                                                        " e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_string(output_resolution,        "-1x-1",        "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " input image resolution.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                                                        " If you want to change the initial scale, you actually want to multiply the"
                                                        " `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number,              1,              "Number of scales to average.");
// OpenPose Rendering
DEFINE_int32(part_to_show,              19,             "Prediction channel to visualize (default: 0). 0 for all the body parts, 1-18 for each body"
                                                        " part heat map, 19 for the background heat map, 20 for all the body part heat maps"
                                                        " together, 21 for all the PAFs, 22-40 for each body part pair PAF");
DEFINE_bool(disable_blending,           false,          "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
                                                        " background, instead of being rendered into the original image. Related: `part_to_show`,"
                                                        " `alpha_pose`, and `alpha_pose`.");
DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                                        " more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
" heatmap, 0 will only show the frame. Only valid for GPU rendering.");



std::map<unsigned int, std::string> bodypartMap;
bool blendOrigFrame;
cv::Size netInputSize;
int numScales;
double scaleGap;
cv::Size outputSize;
cv::Size netOutputSize;
std::string modelFolder;
unsigned int numGpuStart;
op::PoseModel poseModel;
double alphaPose;
int loggingLevel;
std::chrono::high_resolution_clock::time_point start, end;

template <typename T>
T getParam(const ros::NodeHandle& nh, const std::string& param_name, T default_value)
{
    T value;
    if (nh.hasParam(param_name))
    {
        nh.getParam(param_name, value);
    }
    else
    {
        ROS_WARN_STREAM("Parameter '" << param_name << "' not found, defaults to '" << default_value << "'");
        value = default_value;
    }
    return value;
}

op::PoseModel stringToPoseModel(const std::string& pose_model_string)
{
    if (pose_model_string == "COCO")
        return op::PoseModel::COCO_18;
    else if (pose_model_string == "MPI")
        return op::PoseModel::MPI_15;
    else if (pose_model_string == "MPI_4_layers")
        return op::PoseModel::MPI_15_4;
    else
    {
        ROS_ERROR("String does not correspond to any model (COCO, MPI, MPI_4_layers)");
        return op::PoseModel::COCO_18;
    }
}

std::map<unsigned int, std::string> getBodyPartMapFromPoseModel(const op::PoseModel& pose_model)
{
    if (pose_model == op::PoseModel::COCO_18)
    {
        // return op::op::POSE_COCO_BODY_PARTS;
        return op::getPoseBodyPartMapping(op::PoseModel::COCO_18);
    }
    else if (pose_model == op::PoseModel::MPI_15 || pose_model == op::PoseModel::MPI_15_4)
    {
        // return op::POSE_MPI_BODY_PARTS;
        return op::getPoseBodyPartMapping(op::PoseModel::MPI_15);
    }
    else
    {
        ROS_FATAL("Invalid pose model, not map present");
        exit(1);
    }
}


std::shared_ptr<op::CvMatToOpInput> openPoseCvMatToOpInput;
std::shared_ptr<op::CvMatToOpOutput> openPoseCvMatToOpOutput;
std::shared_ptr<op::PoseExtractorCaffe> openPosePoseExtractorCaffe;
std::shared_ptr<op::PoseRenderer> openPosePoseRenderer;
std::shared_ptr<op::OpOutputToCvMat> openPoseOpOutputToCvMat;

bool initOpenPose(){
    ROS_INFO("[Init OpenPose] ...");

    openPosePoseExtractorCaffe = std::shared_ptr<op::PoseExtractorCaffe>(
            new op::PoseExtractorCaffe(poseModel, modelFolder, 0)
    );
    

    op::ConfigureLog::setPriorityThreshold((op::Priority)loggingLevel);


    openPosePoseRenderer = std::shared_ptr<op::PoseRenderer>(
            new op::PoseGpuRenderer(poseModel, nullptr, (float)FLAGS_render_threshold, blendOrigFrame, (float)alphaPose)
    );
    
    /* Initialize everything */
    openPosePoseExtractorCaffe->initializationOnThread();
    openPosePoseRenderer->initializationOnThread();

    return true;
}

void initResponse(openpose_ros::PersonResponse& res){
    for (auto i = 0; i < 1; ++i) {
        openpose_ros::PersonDetection person;
        person.avgConfidence = NAN;
        for (auto j = 0; i < 18; ++i) {
            openpose_ros::Bodypart part;
            part.confidence = NAN;
            part.name = NAN;
            part.x = NAN;
            part.y = NAN;
            person.bodyparts.push_back(part);
        }
        res.detections.push_back(person);
    }
}

bool detectPosesCallback(openpose_ros::PersonRequest& req, openpose_ros::PersonResponse& res){
    // init response
    initResponse(res);

    ROS_INFO("[Called] ######## Start ########");
    start = std::chrono::high_resolution_clock::now();
    // Convert ROS message to opencv image
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(req.image, req.image.encoding);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return false;
    }
    cv::Mat imageCV = cv_ptr->image;
    if(imageCV.empty()) {
        op::error("Could not open or find the image");
        return false;
    }

    // Pose Estimation

    const op::Point<int> imageSize{imageCV.cols, imageCV.rows};

    double scaleInputToOutput;
    //op::Array<float> outputArray;
    std::vector<double> scaleInputToNetInputs;
	std::vector<op::Point<int>> netInputSizes;
	op::Point<int> outputResolution;

	 // outputSize
    const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
    // netInputSize
    const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");

    op::ScaleAndSizeExtractor scaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);
	op::CvMatToOpInput cvMatToOpInput;
	op::CvMatToOpOutput cvMatToOpOutput;
	op::OpOutputToCvMat opOutputToCvMat;
	

    std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution) = scaleAndSizeExtractor.extract(imageSize);

    const auto netInputArray = cvMatToOpInput.createArray(imageCV, scaleInputToNetInputs, netInputSizes);
    auto outputArray = cvMatToOpOutput.createArray(imageCV, scaleInputToOutput, outputResolution);
  
    openPosePoseExtractorCaffe->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
    const auto poseKeyPoints = openPosePoseExtractorCaffe->getPoseKeypoints();
    const auto scaleNetToOutput = openPosePoseExtractorCaffe->getScaleNetToOutput();

   
    openPosePoseRenderer->renderPose(outputArray, poseKeyPoints, scaleInputToOutput, scaleNetToOutput);
  
    auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);

    // Show Results
    /*
    const cv::Size windowedSize = outputSize;
    op::FrameDisplayer frameDisplayer{windowedSize, "OpenPose ROS Wrapper - DEBUG Window"};
    frameDisplayer.displayFrame(outputImage, 1);
    */

    // Prepare Response Message
    if (!poseKeyPoints.empty() && poseKeyPoints.getNumberDimensions() != 3)
    {
        ROS_ERROR("poseKeyPoints: %d != 3", (int) outputArray.getNumberDimensions());
        return false;
    }

    int persons = poseKeyPoints.getSize(0);
    int bodyparts;
    std::string bodypartdesc;

    // Add Image to response
    sensor_msgs::Image imgMsg = *cv_bridge::CvImage(std_msgs::Header(), "bgr8", outputImage).toImageMsg();
    res.detection_img = imgMsg;

    if(persons){
        ROS_INFO("[Called] People detected: %d", persons);
    }else{
        ROS_WARN("[Called] People detected: %d", persons);
    }
    // clear everything we got so far
    res.detections.clear();
    // Iterate over each detected Person
    for(auto person_count = 0; person_count < persons; person_count++){
        bodyparts = poseKeyPoints.getSize(1);
        float avgConfidence = 0;
        int partTakenIntoAccount = 0;
        openpose_ros::PersonDetection person;

        // Iterate over each bodypart of Person(i)
        for(auto body_count = 0; body_count < bodyparts; body_count++){
            openpose_ros::Bodypart part;
            part.name = bodypartMap[body_count]; // What are we looking at here (bodypart name)
            int index_bodymap = 3*(person_count*bodyparts + body_count);
            part.x = poseKeyPoints[index_bodymap]; // X-Coordinate
            part.y = poseKeyPoints[index_bodymap+1]; // Y-Coordinate
            part.confidence = poseKeyPoints[index_bodymap+2]; // Confidence
            person.bodyparts.push_back(part);
            // Only take into account bodypart with a confidence higher than a certain value for the average confidence calculation
            if(part.confidence > 0.1){
                partTakenIntoAccount++;
                avgConfidence += part.confidence;
            }
        }
        if(partTakenIntoAccount > 0) {
            avgConfidence /= partTakenIntoAccount; // Take the average
        } else{
            avgConfidence = 0.0;
        }

        person.avgConfidence = avgConfidence;
        res.detections.push_back(person);
    }

    end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
    ROS_INFO("[Called] Finished in %ldms", duration);
    ROS_INFO("[Called] #######################");
    return true;
}

int main(int argc, char *argv[])
{
    // google::InitGoogleLogging("openpose_ros_node");
    ros::init(argc, argv, "openpose_ros_node");
    ros::NodeHandle local_nh("~");
    netInputSize = cv::Size(getParam(local_nh, "net_input_width", 656), getParam(local_nh, "net_input_height", 368));
    netOutputSize = cv::Size(getParam(local_nh, "net_output_width", 656), getParam(local_nh, "net_output_height", 368));
    outputSize = cv::Size(getParam(local_nh, "output_width", 1280), getParam(local_nh, "output_height", 720));
    numScales = getParam(local_nh, "num_scales", 1);
    scaleGap = getParam(local_nh, "scale_gap", 0.3);
    numGpuStart = getParam(local_nh, "numGpuStart", 0);
    modelFolder = getParam(local_nh, "modelFolder", std::string("/home/user/Pepper/fall_detection/openpose/models/"));
    poseModel = stringToPoseModel(getParam(local_nh, "poseModel", std::string("COCO")));
    bodypartMap = getBodyPartMapFromPoseModel(poseModel);
    alphaPose = 0.99; // Blending factor (range 0-1) for the body part rendering.
    loggingLevel = 200; // The logging level. Integer in the range [0, 255]. 0 -> Everything, 255 -> Nothing
    blendOrigFrame = false;

    if(!initOpenPose()){
        ROS_ERROR("[Init OpenPose] ERROR");
        return 0;
    }

    ROS_INFO("[Init OpenPose] SUCCESS");

    ros::NodeHandle nh;
    ros::ServiceServer service = nh.advertiseService("detect_poses", detectPosesCallback);

    ros::spin();

    return 0;
}

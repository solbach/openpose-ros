#define USE_CAFFE
#include <gflags/gflags.h>
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

std::map<unsigned int, std::string> g_bodypart_map;
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
        return op::POSE_COCO_BODY_PARTS;
    }
    else if (pose_model == op::PoseModel::MPI_15 || pose_model == op::PoseModel::MPI_15_4)
    {
        return op::POSE_MPI_BODY_PARTS;
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

    /* Prepare all classes */
    openPoseCvMatToOpInput = std::shared_ptr<op::CvMatToOpInput>(
            new op::CvMatToOpInput(netInputSize, numScales, (float)scaleGap)
    );

    openPoseCvMatToOpOutput = std::shared_ptr<op::CvMatToOpOutput>(
            new op::CvMatToOpOutput(outputSize)
    );

    openPosePoseExtractorCaffe = std::shared_ptr<op::PoseExtractorCaffe>(
            new op::PoseExtractorCaffe(netInputSize, netOutputSize, outputSize, numScales, (float)scaleGap, poseModel,
                                       modelFolder, 0)
    );

    op::ConfigureLog::setPriorityThreshold((op::Priority)loggingLevel);

    openPosePoseRenderer = std::shared_ptr<op::PoseRenderer>(
            new op::PoseRenderer(netOutputSize, outputSize, poseModel, nullptr, (float)alphaPose)
    );

    openPoseOpOutputToCvMat = std::shared_ptr<op::OpOutputToCvMat>(
            new op::OpOutputToCvMat(outputSize)
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
        //res.result = -1;
        return false;
    }

    // Pose Estimation
    const auto netInputArray = openPoseCvMatToOpInput->format(imageCV);
    double scaleInputToOutput;
    op::Array<float> outputArray;
    std::tie(scaleInputToOutput, outputArray) = openPoseCvMatToOpOutput->format(imageCV);
    openPosePoseExtractorCaffe->forwardPass(netInputArray, imageCV.size());
    const auto poseKeyPoints = openPosePoseExtractorCaffe->getPoseKeyPoints();
    openPosePoseRenderer->renderPose(outputArray, poseKeyPoints);
    auto outputImage = openPoseOpOutputToCvMat->formatToCvMat(outputArray);

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

    if(persons){
        ROS_INFO("[Called] People detected: %d", persons);
        // Add Image to response
        sensor_msgs::Image imgMsg = *cv_bridge::CvImage(std_msgs::Header(), "bgr8", outputImage).toImageMsg();
        res.detection_img = imgMsg;
    }else{
        ROS_WARN("[Called] People detected: %d", persons);
    }
    // Iterate over each detected Person
    for(auto person_count = 0; person_count < persons; person_count++){
        res.detections.clear();
        bodyparts = poseKeyPoints.getSize(1);
        float avgConfidence = 0;
        int partTakenIntoAccount = 0;
        openpose_ros::PersonDetection person;

        // Iterate over each bodypart of Person(i)
        for(auto body_count = 0; body_count < bodyparts; body_count++){
            openpose_ros::Bodypart part;
            part.name = g_bodypart_map[body_count]; // What are we looking at here (bodypart name)
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
    google::InitGoogleLogging("openpose_ros_node");
    ros::init(argc, argv, "openpose_ros_node");
    ros::NodeHandle local_nh("~");
    netInputSize = cv::Size(getParam(local_nh, "net_input_width", 656), getParam(local_nh, "net_input_height", 368));
    netOutputSize = cv::Size(getParam(local_nh, "net_output_width", 656), getParam(local_nh, "net_output_height", 368));
    outputSize = cv::Size(getParam(local_nh, "output_width", 1280), getParam(local_nh, "output_height", 720));
    numScales = getParam(local_nh, "num_scales", 1);
    scaleGap = getParam(local_nh, "scale_gap", 0.3);
    numGpuStart = getParam(local_nh, "numGpuStart", 0);
    modelFolder = getParam(local_nh, "modelFolder", std::string("/home/markus/git/jtl/Fallen Person/CMU-OpenPose/openpose/models/"));
    poseModel = stringToPoseModel(getParam(local_nh, "poseModel", std::string("COCO")));
    g_bodypart_map = getBodyPartMapFromPoseModel(poseModel);
    alphaPose = 0.6; // Blending factor (range 0-1) for the body part rendering.
    loggingLevel = 200; // The logging level. Integer in the range [0, 255]. 0 -> Everything, 255 -> Nothing

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

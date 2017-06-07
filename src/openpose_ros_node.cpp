// ------------------------- OpenPose Library Tutorial - Pose - Example 1 - Extract from Image -------------------------
// This first example shows the user how to:
    // 1. Load an image (`filestream` module)
    // 2. Extract the pose of that image (`pose` module)
    // 3. Render the pose on a resized copy of the input image (`pose` module)
    // 4. Display the rendered pose (`gui` module)
// In addition to the previous OpenPose modules, we also need to use:
    // 1. `core` module: for the Array<float> class that the `pose` module needs
    // 2. `utilities` module: for the error & logging functions, i.e. op::error & op::log respectively

#define USE_CAFFE
// 3rdpary depencencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging, CHECK, CHECK_EQ, LOG, VLOG, ...
// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

#include "ros/ros.h"
#include <std_srvs/Empty.h>
#include <ros/node_handle.h>
#include <ros/service_server.h>
#include <ros/init.h>
#include <cv_bridge/cv_bridge.h>
#include <openpose_ros/Person.h>
#include <chrono>

std::map<unsigned int, std::string> g_bodypart_map;
cv::Size g_net_input_size;
int g_num_scales;
double g_scale_gap;
cv::Size output_size;
cv::Size net_output_size;
std::string model_folder;
unsigned int num_gpu_start;
op::PoseModel pose_model;
double alpha_pose;
int logging_level;
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
            new op::CvMatToOpInput(g_net_input_size, g_num_scales, (float)g_scale_gap)
    );

    openPoseCvMatToOpOutput = std::shared_ptr<op::CvMatToOpOutput>(
            new op::CvMatToOpOutput(output_size)
    );

    openPosePoseExtractorCaffe = std::shared_ptr<op::PoseExtractorCaffe>(
            new op::PoseExtractorCaffe(g_net_input_size, net_output_size, output_size, g_num_scales, (float)g_scale_gap, pose_model,
                                       model_folder, 0)
    );

    op::ConfigureLog::setPriorityThreshold((op::Priority)logging_level);

    openPosePoseRenderer = std::shared_ptr<op::PoseRenderer>(
            new op::PoseRenderer(net_output_size, output_size, pose_model, nullptr, (float)alpha_pose)
    );

    openPoseOpOutputToCvMat = std::shared_ptr<op::OpOutputToCvMat>(
            new op::OpOutputToCvMat(output_size)
    );

    /* Initialize everything */
    openPosePoseExtractorCaffe->initializationOnThread();
    openPosePoseRenderer->initializationOnThread();

    return true;
}


bool detectPosesCallback(openpose_ros::PersonRequest& req, openpose_ros::PersonResponse& res){

    ROS_INFO("[Called] ...");
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
    //cv::Mat imageCV = cv::imread("/home/markus/git/jtl/Fallen Person/CMU-OpenPose/openpose/examples/fallen/fallen_test.jpg", CV_LOAD_IMAGE_COLOR);

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
    const cv::Size windowedSize = output_size;
    op::FrameDisplayer frameDisplayer{windowedSize, "OpenPose ROS Wrapper - DEBUG Window"};
    frameDisplayer.displayFrame(outputImage, 1);


    // Prepare Response Message
    if (!poseKeyPoints.empty() && poseKeyPoints.getNumberDimensions() != 3)
    {
        ROS_ERROR("poseKeyPoints: %d != 3", (int) outputArray.getNumberDimensions());
        return false;
    }

    int persons = poseKeyPoints.getSize(0);
    int bodyparts;
    std::string bodypartdesc;


    for(auto person_count = 0; person_count < persons; person_count++){
        bodyparts = poseKeyPoints.getSize(1);
        ROS_WARN("BODYPARTS DETECTED: %d", bodyparts);

        for(auto body_count = 0; body_count < bodyparts; body_count++){
            bodypartdesc = g_bodypart_map[body_count];

            int index_bodymap = 3*(person_count*bodyparts + body_count);
            int xCoordinate = poseKeyPoints[index_bodymap]; // X-Coordinate
            int yCoordinate = poseKeyPoints[index_bodymap+1]; // Y-Coordinate
            double confidence = poseKeyPoints[index_bodymap+2]; // Confidence

            std::cout << "PART: " << bodypartdesc << std::endl;
            std::cout << "\t x: " << xCoordinate << ", y: " << yCoordinate << std::endl;
            std::cout << "\t confidence: " << confidence << std::endl;
        }
    }


    end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
    ROS_INFO("[Called] Finished in %ld ms", duration);

    //res.detections(0)->  = 5;
    return true;
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "openpose_ros");
    ros::NodeHandle local_nh("~");
    g_net_input_size = cv::Size(getParam(local_nh, "net_input_width", 656), getParam(local_nh, "net_input_height", 368));
    net_output_size = cv::Size(getParam(local_nh, "net_output_width", 656), getParam(local_nh, "net_output_height", 368));
    output_size = cv::Size(getParam(local_nh, "output_width", 1280), getParam(local_nh, "output_height", 720));
    g_num_scales = getParam(local_nh, "num_scales", 1);
    g_scale_gap = getParam(local_nh, "scale_gap", 0.3);
    num_gpu_start = getParam(local_nh, "num_gpu_start", 0);
    model_folder = getParam(local_nh, "model_folder", std::string("/home/markus/git/jtl/Fallen Person/CMU-OpenPose/openpose/models/"));
    pose_model = stringToPoseModel(getParam(local_nh, "pose_model", std::string("COCO")));
    g_bodypart_map = getBodyPartMapFromPoseModel(pose_model);
    alpha_pose = 0.6; // Blending factor (range 0-1) for the body part rendering.
    logging_level = 200; // The logging level. Integer in the range [0, 255]. 0 -> Everything, 255 -> Nothing

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

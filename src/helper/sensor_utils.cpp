#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>

namespace py = pybind11;

/**
 * @brief Projects 3D LiDAR points onto a 2D top-down occupancy grid.
 *
 * Converts a set of 3D points (typically from a LiDAR sensor) into a
 * discretised bird’s-eye-view (BEV) grid representation. Each valid
 * point is mapped to a corresponding grid cell and marked as occupied.
 *
 * @param points Input NumPy array of shape [N, 4], where each point
 *        consists of (x, y, z, intensity).
 * @param grid_size Resolution of the output square grid (grid_size x grid_size).
 * @param range_max Maximum sensing range (metres) used to normalise coordinates.
 *
 * @return A 2D NumPy array (uint8) representing the occupancy grid,
 *         where occupied cells are set to 255 and empty cells to 0.
 */
py::array_t<uint8_t> project_to_2d_points(py::array_t<float> points, int grid_size, float range_max) {
    auto buf = points.request();
    float* ptr = static_cast<float*>(buf.ptr);
    int num_points = buf.shape[0];

    // Create an empty grid (initialised to zero)
    auto grid = py::array_t<uint8_t>({grid_size, grid_size});
    auto grid_buf = grid.request();
    uint8_t* grid_ptr = static_cast<uint8_t*>(grid_buf.ptr);
    std::memset(grid_ptr, 0, grid_size * grid_size * sizeof(uint8_t));

    // Resolution per grid cell (metres per pixel)
    float res = (2.0f * range_max) / grid_size;

    for (int i = 0; i < num_points; ++i) {
        float x = ptr[i * 4 + 0]; ///< Forward axis
        float y = ptr[i * 4 + 1]; ///< Lateral axis

        // Map coordinates to grid indices
        int gi = static_cast<int>((x + range_max) / res);
        int gj = static_cast<int>((y + range_max) / res);

        // Mark valid indices as occupied
        if (gi >= 0 && gi < grid_size && gj >= 0 && gj < grid_size) {
            grid_ptr[gi * grid_size + gj] = 255; // Mark as occupied
        }
    }

    // std::cout << "Projected " << num_points << " points to 2D grid." << std::endl;
    return grid;
}

/**
 * @brief Processes a raw CARLA camera buffer into an RGB image.
 *
 * Converts an input BGRA image buffer into RGB format and resizes it
 * to a target resolution suitable for downstream tasks such as
 * reinforcement learning or perception pipelines.
 *
 * @param input Input NumPy array representing the raw camera buffer
 *        with shape [H, W, 4] (BGRA format).
 * @param target_w Desired output width in pixels.
 * @param target_h Desired output height in pixels.
 *
 * @return A NumPy array (uint8) of shape [target_h, target_w, 3]
 *         containing the processed RGB image.
 *
 * @throws std::runtime_error if the input buffer is invalid or empty.
 */
py::array_t<uint8_t> process_camera_buffer(py::array_t<uint8_t> input, int target_w, int target_h) {
    auto buf = input.request();

    // Validate input buffer
    if (buf.size == 0 || buf.ndim < 2) {
        throw std::runtime_error("Camera input buffer is empty or invalid!");
    }
    
    // CARLA raw_data is [H, W, 4] (BGRA)
    int h = buf.shape[0];
    int w = buf.shape[1];

    // Create OpenCV matrix from raw buffer (BGRA format)
    cv::Mat img(h, w, CV_8UC4, (void*)buf.ptr);

    if (img.empty()) {
        throw std::runtime_error("Could not create OpenCV Mat from buffer.");
    }

    // Convert BGRA to RGB
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGRA2RGB);

    // Resize to target resolution
    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(target_w, target_h), 0, 0, cv::INTER_AREA);

    // Copy result into NumPy array
    auto result = py::array_t<uint8_t>({target_h, target_w, 3});
    std::memcpy(result.request().ptr, resized_img.data, target_h * target_w * 3);

    // std::cout << "Processed camera buffer: original (" << w << "x" << h << ") -> resized ("
    //     << target_w << "x" << target_h << ")" << std::endl;
    return result;
}

/**
 * @brief Renders a combined visualisation of camera and LiDAR data.
 *
 * Displays a side-by-side view consisting of:
 * - Left: RGB camera feed (converted from BGRA).
 * - Right: LiDAR bird’s-eye-view (BEV) projection with distance-based colouring.
 *
 * LiDAR points are colour-mapped based on their Euclidean distance from
 * the sensor, producing a gradient (e.g. red for near, green/blue for far).
 *
 * @param camera_input Raw camera buffer as a NumPy array [H, W, 4] (BGRA).
 * @param lidar_points LiDAR point cloud as a NumPy array [N, 4].
 * @param range_max Maximum LiDAR sensing range (metres).
 */
void draw_sensors(py::array_t<uint8_t> camera_input, py::array_t<float> lidar_points, float range_max) {
    auto cam_buf = camera_input.request();
    int cam_h = cam_buf.shape[0];
    int cam_w = cam_buf.shape[1];

    // Convert camera input from BGRA to BGR for OpenCV display
    cv::Mat raw_cam(cam_h, cam_w, CV_8UC4, (void*)cam_buf.ptr);
    cv::Mat left_view;
    cv::cvtColor(raw_cam, left_view, cv::COLOR_BGRA2BGR);

    int width = 640;
    int height = 480;

    cv::resize(left_view, left_view, cv::Size(width, height));

    // Initialise LiDAR visualisation canvas
    cv::Mat right_view = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
    auto lidar_buf = lidar_points.request();
    float* ptr = static_cast<float*>(lidar_buf.ptr);
    
    // Scaling factor for mapping coordinates to pixels
    float scale = static_cast<float>(height) / (2.0f * range_max);

    for (int i = 0; i < lidar_buf.shape[0]; ++i) {
        float x = ptr[i * 4 + 0]; ///< Forward axis
        float y = ptr[i * 4 + 1]; ///< Lateral axis

        // Compute Euclidean distance from sensor origin: sqrt(x^2 + y^2)
        float dist = std::sqrt(x * x + y * y);
        
        // Normalise distance to [0, 1]
        float normalised_dist = std::min(1.0f, dist / range_max);

        // Map distance to HSV hue (colour gradient)
        float hue = normalised_dist * 120.0f; // 0 is Red, 120 is Green (Standard HSV)

        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(static_cast<int>(hue), 255, 255));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cv::Scalar point_color = cv::Scalar(bgr.data[0], bgr.data[1], bgr.data[2]);

        // Convert to centred BEV coordinates
        int px = width / 2 + static_cast<int>(y * scale); 
        int py = height / 2 - static_cast<int>(x * scale);

        if (px >= 0 && px < width && py >= 0 && py < height) {
            // Use the calculated distance-based color instead of static Green
            cv::circle(right_view, cv::Point(px, py), 0.5, point_color, -1);
        }
    }

    // Create combined display canvas
    cv::Mat canvas = cv::Mat::zeros(height, width * 2, CV_8UC3);

    left_view.copyTo(canvas(cv::Rect(0, 0, width, height)));
    right_view.copyTo(canvas(cv::Rect(width, 0, width, height)));

    // Annotate views
    cv::putText(canvas, "CAMERA", cv::Point(20, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(canvas, "LIDAR (BEV)", cv::Point(660, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    // Display window
    cv::imshow("Agent Perception Hub", canvas);
    cv::waitKey(1);
}

/**
 * @brief Python module definition for sensor utilities.
 *
 * Exposes C++ implementations of sensor processing functions to Python
 * via Pybind11. This includes LiDAR projection, camera preprocessing,
 * and real-time visualisation utilities.
 */
PYBIND11_MODULE(sensor_utils, m) {
    m.def("project_to_2d_points", &project_to_2d_points,
        "Project LiDAR to 2D grid");
    m.def("process_camera_buffer", &process_camera_buffer,
        "Convert BGRA to RGB and resize");
    m.def("draw_sensors", &draw_sensors,
        "Draw sensors using OpenCV C++ windows");
}
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    // Load the image
    CommandLineParser parser(argc, argv, "{@input | card.png | input image}");
    Mat src = imread(parser.get<String>("@input"));

    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    // Show source image
    imshow("Source Image", src);
    
    for (int i = 0; i < src.rows; i++) 
    {
        for (int j = 0; j < src.cols; j++) 
        {
            if (src.at<Vec3b>(i, j) == Vec3b(255, 255, 255))
            {
                src.at<Vec3b>(i, j)[0] = 0;
                src.at<Vec3b>(i, j)[1] = 0;
                src.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }
    // Show output image
    imshow("Black Background Image", src);
    
    Mat kernel = (Mat_<float>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1);

    Mat imgLaplacian;
    
    filter2D(src, imgLaplacian, CV_32F, kernel);
    
    Mat sharp;
    
    src.convertTo(sharp, CV_32F);
    
    Mat imgResult = sharp - imgLaplacian;
    
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
   
    imshow("Laplace Filtered Image", imgLaplacian);
    imshow("New Sharped Image", imgResult);
    
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("Binary Image", bw);
    
    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);

    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    imshow("Distance Transform Image", dist);
    
    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
    
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    imshow("Peaks", dist);
    

    Mat dist_8u;
    
    dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    
    vector<vector<Point> > contours;
    
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    Mat markers = Mat::zeros(dist.size(), CV_16U);
   
    //  Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
    }
    // Draw the background marker

    circle(markers, Point(5, 5), 3, Scalar(255), -1);

    imshow("Markers", markers * 10000);

    markers.convertTo(markers, CV_32S);

    // Perform the watershed algorithm
    watershed(imgResult, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);

    bitwise_not(mark, mark);
    
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<Vec3b>(i, j) = colors[index - 1];
            }
        }
    }
    
    // Visualize the final image
    imshow("Final Result", dst);
    
    waitKey();
    
    return 0;
}
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <boost/gil.hpp>
#include <boost/gil/extension/dynamic_image/any_image.hpp>
#include <boost/gil/extension/io/jpeg.hpp>
#include <boost/mpl/vector.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace boost::gil;

struct Pixel 
{
    int x;
    int y;
    
    Pixel(int x, int y, gray8c_view_t *src): x(x), y(y), src(src) {}

    gray8c_view_t *src;
    
    const double sigma = 2.5;
    const double h = 2.5;

    //Member of equation 6
    double gaussian_kernel()
    {
        return 1 / (2*M_PI*sigma*sigma) * std::exp(-((x*x)+(y*y))/(2*sigma*sigma));
    }
    
    //Member of equation 6
    double weighted_difference(Pixel Y)
    {
        double sum = 0;
        std::mutex m;
        //Integrate over t for each pixel pair (x,y)
        hpx::for_loop(hpx::parallel::execution::par, 0, src->width(), [&](int t_x){
        hpx::for_loop(hpx::parallel::execution::par, 0, src->height(), [&](int t_y){
            //It was ambiguous whether we should normalize t around the X or the Y variable of integration
            //Point t(t_x - X.x, t_y - X.y, src);
            Pixel t(t_x - Y.x, t_y - Y.y, src);
            double diff = ((*src)(x + t.x, y + t.y) - (*src)(Y.x + t.y, Y.y + t.y));

            m.lock();
            sum += t.gaussian_kernel() * diff * diff;
            m.unlock();

        });});

        return sum;
    }

    //Member of equation 6
    double normalizing_factor()
    {
        return 2.5; //placeholder
    }

    //Implemenation of equation 6
    double nonlocal_deblur()
    {
        
        double sum = 0;
        std::mutex m;
        //Integrate over y for each pixel x
        hpx::for_loop(hpx::parallel::execution::par, 0, src->width(), [&](int y_x){
        hpx::for_loop(hpx::parallel::execution::par, 0, src->height(), [&](int y_y){
            
            m.lock();
            sum += std::exp(-1 * this->weighted_difference({y_x,y_y,src})/(h*h)) * (*src)(y_x, y_y);
            m.unlock();
            
        });});

        return (1./normalizing_factor()) * sum;
    
   }

   };

double init_deblur(int x, int y, gray8c_view_t *src)
{
    Pixel p(x,y,&src);
    return p.nonlocal_deblur();
}

//Implementation of equation 10
gray8_image_t deblur_itr(const gray8c_view_t& src)
{
    gray8_image_t new_img(src.width(),src.height());

    std::vector<hpx::future<double>> deblur_values;   
    //Iterate over the image domain, x, ie preform deblurring at each pixel
    for (int x_x=0; x_x<src.width(); x_x++)
    {
        for (int x_y=1; x_y<src.height(); x_y++)
        {       
            deblur_values.push_back(
                hpx::async(init_deblur, x_x, x_y, &src);
            );
        }
    }

    hpx::when_all(deblur_values).then([&](auto&& f)
    {
        auto futures = f.get();
        for(int i = 0; i < futures.size(); i++)
        {
            //since the pixel futures were loaded into a 1 d array, we have to extract their x and y
            //coordinates from their indicies in the future array
            int x = i / src.height();
            int y = i % src.height(); 
            new_img(x,y) = futures[i].get();
        }
    }
    return new_img;
}

//Iteratively deblurs image using equation 10 and a given number of iterations
gray8_image_t deblur_img(gray8c_view_t src, int iterations)
{   
   
    for(int i = 0; i < iterations; i++)
    {      
        gray8_image_t new_img = deblur_itr(src);
        src = new_img;
    }
    
    return src;
}

int main()
{
    using my_img_types = boost::mpl::vector<gray8_image_t, gray16_image_t, rgb8_image_t, rgb16_image_t>;
    any_image<my_img_types> src;
    read_image("Lenna.jpg", src, jpeg_tag());
    
    //here is the new deblurred image after 10 iterations
    gray8_image_t deblurred_img = deblur_img(src, 10);

    return 0;
}

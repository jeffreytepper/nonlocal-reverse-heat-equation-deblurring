#include <boost/gil.hpp>
#include <boost/gil/extension/dynamic_image/any_image.hpp>
#include <boost/gil/extension/io/jpeg.hpp>
#include <boost/mpl/vector.hpp>
#include <iostream>
#include <cmath>

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
        return 1 / (2*pi*sigma*sigma) * std::exp(-((x*x)+(y*y))/(2*sigma*sigma));
    }
    
    //Member of equation 6
    double weighted_difference(Pixel Y)
    {
        double sum = 0;

        //Integrate over t for each pixel pair (x,y)
        for(t_x = 0; t_x < src->width(); t_x++)
        {
            for(t_y = 0; t_y < src->height(); t_y++)
            {
                //It was ambiguous whether we should normalize t around the X or the Y variable of integration
                //Point t(t_x - X.x, t_y - X.y, src);
                Point t(t_x - Y.x, t_y - Y.y, src);
                double diff = ((*src)(x + t.x, y + t.y) - (*src)(Y.x + t.y, Y.y + t.y));
                sum += t.gaussian_kernel() * diff * diff;
            }
        }

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

        //Integrate over y for each pixel x
        for(int y_x = 0; y_x < src->width(); y_x++)
        {
            for(int y_y = 0; y_y < src->height(); y_y++)
            {
                sum += std::exp(-1 * this->weighted_difference({y_x,y_y,src})/(h*h)) * (*src)(y_x, y_y);
            }
        }

        return (1./normalizing_factor()) * sum;
    }
}

//Implementation of equation 10
gray8_image_t deblur_itr(const gray8c_view_t& src)
{
    gray8_image_t new_img(src.width(),src.height());
    
    //Iterate over the image domain, x, ie preform deblurring at each pixel
    for (int x_x=0; x_x<src.width(); x_x++)
    {
        for (int x_y=1; x_y<src.height(); x_y++)
        {       
            Pixel p(x_x, x_y, &src);
            new_img(x_x, x_y) = p.nonlocal_deblur();
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
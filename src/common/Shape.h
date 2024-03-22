

#ifndef __ZJV_SHAPE__
#define __ZJV_SHAPE__


namespace ZJVIDEO
{
    template <typename _Tp> class Point_
    {
    public:
        Point_(): x(0), y(0) {};
        Point_(_Tp x, _Tp y): x(x), y(y) {};
        _Tp x, y;
    };
    typedef Point_<int> Point;
    typedef Point_<float> Point2f;

    template <typename _Tp> class Size_
    {
    public:
        Size_(): width(0), height(0) {};
        Size_(int width, int height): width(width), height(height) {};
        int width, height;
    };
    typedef Size_<int> Size;
    typedef Size_<float> Size2f;

    template <typename _Tp> class Rect_
    {
    public:
        Rect_(): x(0), y(0), width(0), height(0) {};
        Rect_(_Tp x, _Tp y, _Tp width, _Tp height): x(x), y(y), width(width), height(height) {};
        Rect_(Point_<_Tp> p, Size_<_Tp> s): x(p.x), y(p.y), width(s.width), height(s.height) {};
        _Tp x, y, width, height;
        _Tp area() const
        {
            return width * height;
        }
        Point_<_Tp> tl() const
        {
            return Point_<_Tp>(x, y);
        }
        Point_<_Tp> br() const
        {
            return Point_<_Tp>(x + width, y + height);
        }
        
    };
    typedef Rect_<int> Rect;
    typedef Rect_<float> Rect2f;
}

#endif
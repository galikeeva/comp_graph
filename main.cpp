#define NULL 0
#define PI 3.1415926f
#define RAY_T_MIN 0.0001f
#define RAY_T_MAX 1.0e30f

#include <cmath>
#include <cstring>
#include <vector>
#include <limits>
#include <unordered_map>
#include <fstream>
#include <iostream>

#include "Bitmap.h"

inline float sqr(float n)
{
	return n * n;
}

struct Vector
{
	float x, y, z;
	Vector(): x(0.0f), y(1.0f), z(0.0f) {}
    Vector(const Vector& v): x(v.x), y(v.y), z(v.z) {}
    Vector(float x, float y, float z): x(x), y(y), z(z) {}
    Vector(float f) : x(f), y(f), z(f) {}
    ~Vector() {}
    inline float length2(){return sqr(x) + sqr(y) + sqr(z);}
    inline float length() {return std::sqrt(length2());}
    float normalize() {float l = length(); *this /= l; return l;}
    Vector normalized() {Vector v(*this); v.normalize(); return v;}
    Vector& operator =(const Vector& v) {x = v.x; y = v.y; z = v.z; return *this;}
    Vector& operator +=(const Vector& v) {x += v.x; y += v.y; z += v.z; return *this;}
    Vector& operator -=(const Vector& v) {x -= v.x; y -= v.y; z -= v.z; return *this;}
    Vector& operator *=(float f) {x *= f; y *= f; z *= f; return *this;}
    Vector& operator /=(float f) {x /= f; y /= f; z /= f; return *this;}
    Vector operator -() const {return Vector(-x, -y, -z);}
    Vector operator +(const Vector& v2) const {return Vector(x + v2.x, y + v2.y, z + v2.z);}
    Vector operator -(const Vector& v2) const {return Vector(x - v2.x, y - v2.y, z - v2.z);}
    Vector operator *(const Vector& v2) const {return Vector(x * v2.x, y * v2.y, z * v2.z);}
    Vector operator *(const float f)  const {return Vector(x * f, y * f, z * f);}
    friend Vector operator *(float f, const Vector& v) {return Vector(f * v.x, f * v.y, f * v.z);}

};
typedef Vector Point;

float dot(Vector v1, Vector v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;}
Vector cross(Vector v1, Vector v2) {
    return Vector(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

struct Vector2
{
	float u, v;
	Vector2(): u(0.0f), v(0.0f) {}
    Vector2(const Vector2 &v): u(v.u),v(v.v) {}
    Vector2(float u, float v): u(u), v(v) {}
    Vector2(float f): u(f), v(f) {}
    ~Vector2() {}
    Vector2& operator =(const Vector2& v){u = v.u; this->v = v.v; return *this;}
    Vector2 operator * (const float &r) const {return Vector2(u * r, v * r);}
    Vector2 operator + (const Vector2 &h) const {return Vector2(u + h.u, v + h.v);}
};

struct Color
{
	float r, g, b;
	Color(): r(0.0f), g(0.0f), b(0.0f) {}
    Color(float l): r(l), g(l), b(l) {}
    Color(float r, float g, float b): r(r), g(g), b(b) {}
    ~Color() {}
    void clamp(float min  = 0.0f, float max  = 1.0f)
    {
        r = std::max(min, std::min(max, r));
        g = std::max(min, std::min(max, g));
        b = std::max(min, std::min(max, b));
    }
    void applyGammaCorrection(float exposure, float gamma)
    {
        r = std::pow(r * exposure, gamma);
        g = std::pow(g * exposure, gamma);
        b = std::pow(b * exposure, gamma);
    }
    Color& operator =(const Color& c) {r = c.r; g = c.g; b = c.b; return *this;}
    Color& operator +=(const Color& c) {r += c.r; g += c.g; b += c.b; return *this;}
    Color& operator *=(const Color& c) {r *= c.r; g *= c.g;b *= c.b;return *this;}
    Color& operator *=(float f) {r *= f; g *= f; b *= f; return *this;}
    Color operator +(const Color& c2) {return Color(r + c2.r, g + c2.g, b + c2.b);}
    Color operator *(const Color& c2) {return Color(r * c2.r, g * c2.g, b * c2.b);}
    Color operator *(float f) {return Color(r * f, g * f, b * f);}
    friend Color operator *(float f, const Color& c) {return Color(f * c.r, f * c.g, f * c.b);}
};

struct Ray
{
	Point origin; // start
	Vector direction;
	float tMax;
	Ray(): origin(0.0f, 0.0f, 0.0f), direction(), tMax(RAY_T_MAX) {}
    Ray(const Ray& r): origin(r.origin), direction(r.direction), tMax(r.tMax) {}
    Ray(const Point& origin, const Vector& direction, float tMax = RAY_T_MAX) : origin(origin), direction(direction), tMax(tMax) {}
    ~Ray() {}
    Ray& operator =(const Ray& r) {origin = r.origin; direction = r.direction; tMax = r.tMax; return *this;}
    Point calculate(float t) {return origin + direction * t;}
};

class Camera
{
public:
    virtual ~Camera() { }
    virtual Ray makeRay(Vector2 point) const = 0;
};

class PerspectiveCamera : public Camera
{
protected:
	Point origin;
	Vector forward;
	Vector up;
	Vector right;
    float h, w;
public:
    PerspectiveCamera(Point origin, Vector target, Vector upguide, float fov, float aspectRatio)
	: origin(origin) {
        forward = (target - origin).normalized();
        right = cross(forward, upguide).normalized();
        up = cross(right, forward);
        h = tan(fov);
        w = h * aspectRatio;
    }
    virtual Ray makeRay(Vector2 point) const {
        Vector direction = forward + point.u * w * right + point.v * h * up;
        return Ray(origin, direction.normalized());
    }
};

class Image
{
protected:
	int width, height;
	Color* data;
public:
    Image(int width, int height): width(width), height(height) {data = new Color[width * height];}
    ~Image() {delete[] data;}
    int getWidth() const {return width;}
    int getHeight() const {return height;}
    Color* getPixel(int x, int y) {return data + (x + y * width);}
    void saveImage(std::string filename, float exposure = 1.0f, float gamma = 2.2f) const{

        Color *data1 = new Color[width * height];
        uint32_t *data2 = new uint32_t[width * height];
        int k = 0;
        for (int y = height - 1; y >= 0; y--){
            for (int x = 0; x < width; x++) {
                Color curColor = data[x + y * width];
                curColor.applyGammaCorrection(exposure, gamma);
                curColor.clamp();
                data1[k] = curColor;
                k++;
            }
        }

        std::ofstream ofs;
    ofs.open("./out.ppm");
    ofs << "P6\n" << width << " " << height << "\n255\n";

        for (uint32_t i = 0; i < uint32_t(height * width); ++i) {
            u_char r = (u_char)(255.0f * data1[i].r);
            u_char g = (u_char)(255.0f * data1[i].g);
            u_char b = (u_char)(255.0f * data1[i].b);
            data2[i] = (u_int(b) << 16) | (u_int(g) << 8) |
                u_int(r);
            ofs << r << g << b;

        }
        ofs.close();

        SaveBMP(filename.c_str(), data2, width, height);

    }
};

class Light
{
public:
    Light(const Vector &p, const float &i) : position(p), intensity(i) {}
    Vector position;
    float intensity;
};

struct Material {
    Material(const Vector &a, const Color &color, const float &spec, const float &ref_i, const float &ref_a)
    : albedo(a), color(color), specular_exponent(spec), ref_ind(ref_i), ref_alb(ref_a) {}
    Material() : albedo(1, 0, 0), color(1, 1, 1), specular_exponent(), ref_ind(), ref_alb() {}
    Vector albedo;
    Color color;
    float specular_exponent;
    float ref_ind;
    float ref_alb;
};

class Sphere
{

public:
    Point centre;
	float radius;
    Material material;
	Sphere(const Point& centre, float radius,
		const Material& material)
	: centre(centre), radius(radius), material(material) {}
	~Sphere() {}
    bool doesIntersect(const Vector &orig, const Vector &dir, Vector &hit, Vector &N, Material &material1,
        float &dist , const std::vector<Sphere> &sh) const{
        // Transform ray so we can consider origin-centred sphere
        Ray localRay = Ray(orig, dir);
        localRay.origin -= centre;
        // Calculate quadratic coefficients
        float a = localRay.direction.length2();
        float b = 2 * dot(localRay.direction, localRay.origin);
        float c = localRay.origin.length2() - sqr(radius);
        // Check whether we intersect
        float discriminant = sqr(b) - 4 * a * c;
        if (discriminant < 0.0f) {
            return false;
        }
        // Find two points of intersection, t1 close and t2 far
        float t1 = (-b - std::sqrt(discriminant)) / (2 * a);
        if (t1 > RAY_T_MIN && t1 < RAY_T_MAX) {
            dist = t1;
            return true;
        }
        float t2 = (-b + std::sqrt(discriminant)) / (2 * a);
        if (t2 > RAY_T_MIN && t2 < RAY_T_MAX) {
            dist = t2;
            return true;
        }
        return false;
    }
};

class Shape
{
public:
	virtual ~Shape() {}
	Point centre;
	Material material;
	virtual Color intersect(const Vector &orig, const Vector &dir, const std::vector <Light> &lights,
        int depth, const std::vector<Sphere> &sh) = 0;
	virtual bool doesIntersect(const Vector &orig, const Vector &dir, Vector &hit, Vector &N,
        Material &material1, float &dist, const std::vector<Sphere> &sh) const = 0;
};

class Plane
{
public:
	Point position;
	Vector normal;
    Color color1;
	Color color2;
	Plane(const Point& position, const Vector& normal,
		const Color& color1 = Color(1.0f, 1.0f, 1.0f), const Color& color2 = Color(1.0f, 1.0f, 1.0f))
		: position(position), normal(normal), color1(color1), color2(color2) {}
    ~Plane() {}
	bool doesIntersect(const Vector &orig, const Vector &dir, Vector &hit, Vector &N, Material &material,
        float &dist, const std::vector<Sphere> &sh) const{
        // First, check if we intersect
        float dDotN = dot(dir, normal);
        if (dDotN == 0.0f) {
            // We just assume the ray is not embedded in the plane
            return false;
        }
        // Find point of intersection
        float t = dot(position - orig, normal) / dDotN;
        if (t <= RAY_T_MIN || t >= RAY_T_MAX) {
            // Outside relevant range
            return false;
        }
        dist = t;
        return true;
    }
};

Vector reflect(const Vector &I, const Vector &N) {
    return I - 2 * dot(I, N) * N;
}

Vector refract(const Vector &I, const Vector &N, const float &refractive_index) { // Snell's law
    float cosi = - std::max(-1.f, std::min(1.f, dot(I, N)));
    float etai = 1, etat = refractive_index;
    Vector n = N;
    if (cosi < 0) { // if the ray is inside the object, swap the indices and invert the normal to get the correct result
        cosi = -cosi;
        std::swap(etai, etat); n = -N;
    }
    float eta = etai / etat;
    float n_cos = 1 - eta * eta*(1 - cosi * cosi);
    return n_cos < 0 ? Vector(1, 0, 0) : I * eta + n * (eta * cosi - sqrtf(n_cos));
}

class ShapeSet : public Shape
{
public:
    ShapeSet() {};
	~ShapeSet() {};
    Color intersect(const Vector &orig, const Vector &dir, const std::vector <Light> &lights,
        int depth, const std::vector<Sphere> &sh)
    {
        Vector hit, N;
        Material material1, tmp;
        float dist;
        if (depth > 4 || !((*this).doesIntersect(orig, dir, hit, N, material1, dist, sh))) {
            return Color(0, 0.4, 0.7);
        }
        Vector reflect_dir = reflect(dir, N).normalized();
        Vector refract_dir = refract(dir, N, material.ref_ind).normalized();
        Vector reflect_orig = dot(reflect_dir, N) < 0 ? hit - N * 1e-5 : hit + N * 1e-5; // offset the original point to avoid occlusion by the object itself
        Vector refract_orig = dot(refract_dir, N) < 0 ? hit - N * 1e-5 : hit + N * 1e-5;
        Color reflect_color = (*this).intersect(reflect_orig, reflect_dir, lights, depth + 1, sh);
        Color refract_color = (*this).intersect(refract_orig, refract_dir, lights, depth + 1, sh);

        float dif_light_ints = 0, spec_light_ints = 0;
        for (size_t i = 0; i < lights.size(); i++){
            Vector l_dir = (lights[i].position - hit).normalized();
            float light_distance = (lights[i].position - hit).normalize();
            Vector shadow_orig = dot(l_dir, N) < 0 ? hit - N * 1e-5 : hit + N * 1e-5; // checking if the point lies in the shadow of the lights[i]
            Vector shadow_pt, v_N;
            float d;
            if ((*this).doesIntersect(shadow_orig, l_dir, shadow_pt, v_N, tmp, d, sh) &&
                ((shadow_pt-shadow_orig).normalize() < light_distance))
                continue;

            dif_light_ints += lights[i].intensity * (std::max(0.f, dot(l_dir, N)));
            spec_light_ints += powf(std::max(0.f, dot(reflect(l_dir, N), dir)), material1.specular_exponent) *
                lights[i].intensity;

        }

        return material1.color * dif_light_ints * material1.albedo.x +
            Color(1., 1., 1.) * spec_light_ints * material1.albedo.y +
            reflect_color * material1.albedo.z + refract_color * material1.ref_alb;
    }
    bool doesIntersect(const Vector &orig, const Vector &dir, Vector &hit, Vector &N, Material &material1,
        float &dist, const std::vector<Sphere> &sh) const {
        float shapes_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < sh.size(); i++) {
            float dist_i;
            if (sh[i].doesIntersect(orig, dir, hit, N, material1, dist_i, sh) && dist_i < shapes_dist) {
                shapes_dist = dist_i;
                hit = orig + dir * dist_i;
                N = (hit - sh[i].centre).normalized();
                material1 = sh[i].material;
            }
        }

        float checkerboard_dist = std::numeric_limits<float>::max();

        Plane floor(Point(0., -1., -2.), Vector(0., 1., 5.),
            Color(.3, .3, .3), Color(.3, .14, .1));
        float d;
        floor.doesIntersect(orig, dir, hit, N, material1, d, sh);
        Vector pt = orig + dir * d;
        if (d > 0  && d < shapes_dist) {
            checkerboard_dist = d;
            hit = pt;
            N = floor.normal;
            material1.color = (int(hit.x + 100) + int(hit.y + 100)) & 1 ? floor.color2 : floor.color1;
        }
        return std::min(shapes_dist, checkerboard_dist) < 1000;
    }
};

void rayTrace(Image& image, Camera* camera, Shape* scene, const std::vector <Light> &lights, const std::vector<Sphere> &sh)
{
	for (int x = 0; x < image.getWidth(); x++) {
		for (int y = 0; y < image.getHeight(); y++) {
			Vector2 screenCoord((2. * x) / image.getWidth() - 1.,
				(-2. * y) / image.getHeight() + 1.);
			Ray ray = camera->makeRay(screenCoord);
			Color* curPixel = image.getPixel(x, y);
			*curPixel = scene->intersect(ray.origin, ray.direction, lights, 0, sh);
		}
	}
}

int main(int argc, char *argv[])
{
	int width = 1920;
	int height = 1080;

	Image image(width, height);
	PerspectiveCamera camera(Point(0., -5., 0.),
		Vector(0., 0.f, 0.), Vector(0., 0., 1.), 25. * PI / 180.,
		(float)width / (float)height);

	ShapeSet scene;
	std::vector <Sphere> sh;

	Material snow_body(Vector(.6, .3, 0.), Color(1., 1., 1.), 50., 1., 0.);

	Sphere sphere_d(Point(0., 1., -.9), 1.2, snow_body);
	sh.push_back(sphere_d);

	Sphere sphere_m(Point(0., 1., .5), .9, snow_body);
	sh.push_back(sphere_m);

	Sphere sphere_u(Point(0., 1., 1.7), .6, snow_body);
    sh.push_back(sphere_u);

    Material snow_hands(Vector(.8, .2, .2), Color(1., 1., 1.), 50., 1., 0.);

	Sphere sphere_rh(Point(1.1, 1., .5), .3, snow_hands);
	sh.push_back(sphere_rh);

	Sphere sphere_lh(Point(-1.1, 1., .5), .3, snow_hands);
	sh.push_back(sphere_lh);

	Material coal(Vector(.5, .5, 0.), Color(0., 0., 0.), 50., 1., 0.);

	Sphere sphere_p1(Point(0., -.1, -1.4), .08, coal);
	sh.push_back(sphere_p1);

	Sphere sphere_p2(Point(0., -.2, -.7), .08, coal);
	sh.push_back(sphere_p2);

	Sphere sphere_p3(Point(0., .17, .0), .08, coal);
	sh.push_back(sphere_p3);

	Sphere sphere_p4(Point(0., .1, 0.7), .08, coal);
	sh.push_back(sphere_p4);

	Sphere sphere_el(Point(-.3, .5, 1.9), .07,
		Material(Vector(.5, .4, .1), Color(0., 0., 0.), 50., 1., 0.));
	sh.push_back(sphere_el);

	Sphere sphere_er(Point(.3, .5, 1.9), .07, coal);
	sh.push_back(sphere_er);

    Sphere sphere_mo(Point(0., .5, 1.5), .08, coal);
	sh.push_back(sphere_mo);

	Sphere sphere_car(Point(0., .4, 1.7), .08,
		Material(Vector(.6, 5., .4), Color(.8, 0., 0.), 1425., 1., 0.));
	sh.push_back(sphere_car);

	Material glass(Vector(0., .5, .1), Color(.6, .7, .8), 125., 1.5, .8);

	Sphere sphere_disco1(Point(1.4, 1., -.4), .5, glass);
	sh.push_back(sphere_disco1);

	Material mirror(Vector(0., 10., .8), Color(1., 1., 1.), 1425., 1., 0.);

	Sphere sphere_disco2(Point(-3.5, 1., 1.4), .8, mirror);
	sh.push_back(sphere_disco2);

	std::vector <Light> lights;
	lights.push_back(Light(Vector(-2, -1, 2), 1.5));
	lights.push_back(Light(Vector(4, -6, 3), 0.8));

	rayTrace(image, &camera, &scene, lights, sh);

	std::unordered_map<std::string, std::string> cmdLineParams;


    for(int i = 0; i < argc; i++) {
        std::string key(argv[i]);
        if(key.size() > 0 && key[0]=='-') {
            if(i != argc - 1) {
                cmdLineParams[key] = argv[i + 1];
                i++;
            }
            else
                cmdLineParams[key] = "";
        }
    }
    std::string outFilePath = "zout.bmp";
    if(cmdLineParams.find("-out") != cmdLineParams.end())
        outFilePath = cmdLineParams["-out"];

    int sceneId = 0;
    if(cmdLineParams.find("-scene") != cmdLineParams.end()) {
        sceneId = atoi(cmdLineParams["-scene"].c_str());
    }
    if(sceneId == 1 || sceneId == 0) {
        image.saveImage(outFilePath);
        return 0;
    } else {
        return 0;
    }
}

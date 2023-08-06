import math
import time


def rbbox_to_corners(rbbox):
    # generate clockwise corners and rotate it clockwise
    # 顺时针方向返回角点位置
    cx, cy, x_d, y_d, angle = rbbox
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    corners_x = [-x_d / 2, -x_d / 2, x_d / 2, x_d / 2]
    corners_y = [-y_d / 2, y_d / 2, y_d / 2, -y_d / 2]
    corners = [0] * 8
    for i in range(4):
        corners[2 *
                i] = a_cos * corners_x[i] + \
                     a_sin * corners_y[i] + cx
        corners[2 * i +
                1] = -a_sin * corners_x[i] + \
                     a_cos * corners_y[i] + cy
    return corners


def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


def line_segment_intersection(pts1, pts2, i, j):
    # pts1, pts2 为corners
    # i j 分别表示第几个交点，取其和其后一个点构成的线段
    # 返回为 tuple(bool, pts) bool=True pts为交点
    A, B, C, D, ret = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    # 叉乘判断方向
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        # 判断方向
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            ret[0] = Dx / DH
            ret[1] = Dy / DH
            return True, ret
    return False, ret


def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    def _cmp(pt, center):
        vx = pt[0] - center[0]
        vy = pt[1] - center[1]
        d = math.sqrt(vx * vx + vy * vy)+1e-5 #防止分母为0
        vx /= d
        vy /= d
        if vy < 0:
            vx = -2 - vx
        return vx

    if num_of_inter > 0:
        center = [0, 0]
        for i in range(num_of_inter):
            center[0] += int_pts[i][0]
            center[1] += int_pts[i][1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        int_pts.sort(key=lambda x: _cmp(x, center))


def area(int_pts, num_of_inter):
    def _trangle_area(a, b, c):
        return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) *
                (b[0] - c[0])) / 2.0

    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            _trangle_area(int_pts[0], int_pts[i + 1],
                          int_pts[i + 2]))
    return area_val

def get_iou(rbbox1,rbbox2):

    corners1 = rbbox_to_corners(rbbox1)
    corners2 = rbbox_to_corners(rbbox2)
    pts, num_pts = [], 0
    for i in range(4):
        point = [corners1[2 * i], corners1[2 * i + 1]]
        if point_in_quadrilateral(point[0], point[1],
                                  corners2):
            num_pts += 1
            pts.append(point)
    for i in range(4):
        point = [corners2[2 * i], corners2[2 * i + 1]]
        if point_in_quadrilateral(point[0], point[1],
                                  corners1):
            num_pts += 1
            pts.append(point)
    for i in range(4):
        for j in range(4):
            ret, point = line_segment_intersection(corners1, corners2, i, j)
            if ret:
                num_pts += 1
                pts.append(point)
    sort_vertex_in_convex_polygon(pts, num_pts)
    intersection = area(pts, num_pts)

    cx1, cy1, dx1, dy1, angle1 = rbbox1
    cx2, cy2, dx2, dy2, angle2 = rbbox2

    union = dx1*dy1 + dx2*dy2 - intersection

    res_iou = intersection / union

    return res_iou


if __name__ == '__main__':
    t1 = time.time()
    for c in range(100):
        rbbox1 = [0, 0, 10, 10, 0] #cx, cy, x_d, y_d, angle
        #rbbox2 = [0, 0, 100, 100, 0]
        rbbox2 = [0.5, 0.5, math.sqrt(2), math.sqrt(2), math.pi/4] #cx, cy, x_d, y_d, angle

        polygon_area = get_iou(rbbox1,rbbox2)

    t2 = time.time()
    print("t_ms=",(t2-t1)*1000)
    print('area: {}'.format(polygon_area))


from scipy.optimize import minimize
import numpy as np

# 주어진 데이터
robot_coords = np.array([[210.253077, 119.379798],
                         [206.136532, -5.775979],
                         [206.040612, -129.310447]])

yolo_coords = np.array([[588.1375, 61.6406],
                        [316.3914, 64.1649],
                        [48.3754, 81.9251]])

# 초기 추정값
initial_guess = np.zeros(6)

# 최적화 문제 정의
def objective(params):
    a, b, c, d, e, f = params
    transform_matrix = np.array([[a, b], [c, d]])
    translation_vector = np.array([e, f])
    yolo_transformed = np.dot(transform_matrix, yolo_coords.T).T + translation_vector
    return np.sum((robot_coords - yolo_transformed) ** 2)

# 최소화
result = minimize(objective, initial_guess, method='L-BFGS-B')

# 최적의 매개변수 출력
transform_matrix = result.x[:4].reshape((2, 2))
translation_vector = result.x[4:]

print("Transform Matrix:")
print(transform_matrix)
print("\nTranslation Vector:")
print(translation_vector)
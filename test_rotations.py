"""
Unit tests for the rotations module
"""



import numpy as np
import rotations



def test_quaternion_from_euler():
    """
    Tests if the function 'quaternion_from_euler' works for a zero rotation.
    """

    assert np.allclose(np.array([1, 0, 0, 0], dtype=float),
                       rotations.quaternion_from_euler(np.array([0, 0, 0], dtype=float)))



# Two (partial) roundtrips between matrices and quaternions ...

def test_matrix_quaternion():
    """
    Tests the functions `quaternion_from_matrix` and
    `matrix_from_quaternion` by creating a random
    rotation matrix (via the functions `matrix_from_quaternion` and
    `quaternion_from_euler`) and checking if the composition of the two
    functions renders the identity.
    """

    # create a random rotation in axis-angle representation
    euler = np.random.normal(loc=0.0, scale=4.0, size=3)

    # convert to a rotation matrix
    known_matrix = rotations.matrix_from_quaternion( \
                   rotations.quaternion_from_euler(euler))

    # assert that the rotation matrix is unitary
    assert np.allclose(np.eye(3), np.dot(known_matrix.T, known_matrix))

    # apply the composition of the two functions
    matrix = rotations.matrix_from_quaternion( \
             rotations.quaternion_from_matrix(known_matrix))

    assert np.allclose(known_matrix, matrix)



def test_quaternion_matrix():
    """
    Tests the functions `matrix_from_quaternion` and `quaternion_from_matrix`
    by creating a random unit quaternion and checking if the composition of
    the two functions renders the identity.
    """

    # creating a random unit quaternion
    known_quaternion = np.random.randn(4)
    known_quaternion /= np.linalg.norm(known_quaternion)
    if known_quaternion[0] < 0:
        known_quaternion *= -1

    # apply the composition of the two functions
    quaternion = rotations.quaternion_from_matrix( \
                 rotations.matrix_from_quaternion(known_quaternion))

    assert np.allclose(known_quaternion, quaternion)



# Two (partial) roundtrips between Euler vectors and quaternions ...

def test_euler_quaternion():
    """
    Tests the functions `quaternion_from_euler` and `euler_from_quaternion`
    by creating a random Euler vector and checking if the composition of
    the two functions renders the identity.
    """

    # create a random axis-angle representation (-2pi < theta <+2pi)
    while True:
        known_euler = np.random.normal(loc=0.0, scale=1.0, size=3)
        if np.linalg.norm(known_euler) < 2*np.pi:
            break

    # apply the composition of the two functions
    euler = rotations.euler_from_quaternion( \
            rotations.quaternion_from_euler(known_euler))

    check = np.allclose(known_euler, euler)

    if not check:
        print("known_euler", known_euler)
        print("euler", euler)

    assert check


def test_quaternion_euler():
    """
    Tests the functions `euler_from_quaternion` and `quaternion_from_euler`
    by creating a random unit quaternion and checking if the composition of the two
    functions renders the identity.
    """

    # creating a random unit quaternion
    known_quaternion = np.random.randn(4)
    known_quaternion /= np.linalg.norm(known_quaternion)

    # apply the composition of the two functions
    quaternion = rotations.quaternion_from_euler( \
                 rotations.euler_from_quaternion(known_quaternion))

    assert np.allclose(known_quaternion, quaternion)




# Two (full) roundtrips between Euler vectors and rotation matrices

def test_euler_matrix():
    """
    Tests the functions `quaternion_from_euler`, `matrix_from_quaternion`,
    `quaternion_from_matrix` and `euler_from_quaternion` by creating a random
    rotation Euler vector and checking if the composition of the four functions
    renders the identity. Additionally, the same is checked for the
    zero (Euler) vector.
    """

    # create a random axis-angle representation (-2pi < theta <+2pi)
    while True:
        known_euler = np.random.normal(loc=0.0, scale=1.0, size=3)
        if np.linalg.norm(known_euler) < 2*np.pi:
            break

    # apply the composition of the two functions
    euler = rotations.euler_from_quaternion( \
            rotations.quaternion_from_matrix( \
            rotations.matrix_from_quaternion( \
            rotations.quaternion_from_euler(known_euler))))

    check = np.allclose(known_euler, euler)

    if not check:
        print("known_euler", known_euler)
        print("euler", euler)

    assert check

    # check additionally if it works for the zero-rotation
    assert np.allclose(rotations.euler_from_quaternion( \
                       rotations.quaternion_from_matrix( \
                       rotations.matrix_from_quaternion( \
                       rotations.quaternion_from_euler(np.zeros((3), dtype=float))))),
                       np.zeros((3), dtype=float))



def test_matrix_euler():
    """
    Tests the functions `quaternion_from_euler`, `matrix_from_quaternion`,
    `quaternion_from_matrix` and `euler_from_quaternion` by creating a random
    rotation Euler vector and checking if the composition of the four functions
    renders the identity. Additionally, the same is checked for the
    zero (Euler) vector.
    """

    # create a random rotation in axis-angle representation
    euler = np.random.normal(loc=0.0, scale=4.0, size=3)

    # convert to a rotation matrix
    known_matrix = rotations.matrix_from_quaternion( \
                   rotations.quaternion_from_euler(euler))

    # assert that the rotation matrix is unitary
    assert np.allclose(np.eye(3), np.dot(known_matrix.T, known_matrix))

    # apply the composition of the two functions
    matrix = rotations.matrix_from_quaternion( \
             rotations.quaternion_from_euler( \
             rotations.euler_from_quaternion( \
             rotations.quaternion_from_matrix(known_matrix))))

    check = np.allclose(known_matrix, matrix)

    if not check:
        print("known_matrix", known_matrix)
        print("matrix", matrix)

    assert check

    # check additionally if it works for the zero-rotation
    assert np.allclose(rotations.matrix_from_quaternion( \
                       rotations.quaternion_from_euler( \
                       rotations.euler_from_quaternion( \
                       rotations.quaternion_from_matrix(np.eye(3, dtype=float))))),
                       np.eye(3, dtype=float))



def test_matrix_from_euler():
    """
    Tests the functions `matrix_from_quaternion` and `quaternion_from_euler` by
    creating a known rotation matrix (rotation arround z-axis by a random angle)
    and comparing the results.
    """

    # rotate around z-axis by a random angle theta (-2pi < theta <+2pi)
    theta = float(np.random.uniform(low=-1.99, high=+1.99, size=1) * np.pi)

    known_matrix = np.array([[+np.cos(theta), -np.sin(theta), 0],
                             [+np.sin(theta), +np.cos(theta), 0],
                             [0, 0, 1]])

    known_euler = theta * np.array([0, 0, 1])

    matrix = rotations.matrix_from_quaternion( \
             rotations.quaternion_from_euler(known_euler))

    assert np.allclose(known_matrix, matrix)

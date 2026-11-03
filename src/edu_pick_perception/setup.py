from setuptools import setup

package_name = "edu_pick_perception"
setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="edu_pick",
    maintainer_email="101551311+karevaros@users.noreply.github.com",
    description="Pick and place 실습 — 물체 인식(깊이 기반) + 가상 카메라",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "object_detector = edu_pick_perception.object_detector:main",
        "fake_camera = edu_pick_perception.fake_camera:main",
    ]},
)

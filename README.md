# taichi-version-update-problem

大家好，最近从 1.13 升级到了 1.30, 发现编译速度慢了好多TAT。

一个简化的例子上传到github了：GitHub - JinliBot7/taichi-version-update-problem

在1.13版本下运行 Energy_function_test.py (import compute_stress_113), gpu和cpu都是编译10秒以下。

在1.30版本下运行 Energy_function_test.py (import compute_stress), gpu和cpu都要编译30秒左右。

compute_stress 和 compute_stress_113 只在104行和113行有差异，是因为更新后向量transpose好像不一样了。

请问应该怎么修改呢？

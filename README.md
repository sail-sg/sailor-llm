
## Evaluation

### Question Answering

<table>
<tr>
    <th>3-shot (EM / F1)</th>
    <th>XQuAD (th)</th>
    <th>TyDi QA (id)</th>
    <th>XQuAD (vi)</th>
</tr>
<tr>
  <td>Qwen1.5-0.5B</td>
  <td>14.19 / 23.35</td>
  <td>20.71 / 32.64</td>
  <td>19.85 / 35.38</td>
</tr>
<tr style="color: #000084;">
  <td>Sailor-0.5B</td>
  <td>15.84 / 27.58</td>
  <td>30.44 / 54.74</td>
  <td>21.13 / 40.57</td>
</tr>
<tr>
  <td>Qwen1.5-1.8B</td>
  <td>27.24 / 43.56</td>
  <td>29.73 / 53.76</td>
  <td>29.17 / 48.15</td>
</tr>
<tr style="color: #000084;">
  <td>Sailor-1.8B</td>
  <td>32.72 / 48.66</td>
  <td>40.88 / 65.37</td>
  <td>34.22 / 53.35</td>
</tr>
<tr>
  <td>Qwen1.5-4B</td>
  <td>34.03 / 53.40</td>
  <td>48.32 / 72.68</td>
  <td>43.71 / 63.86</td>
</tr>
<tr style="color: #000084;">
  <td>Sailor-4B</td>
  <td>46.82 / 63.34</td>
  <td>53.98 / 73.48</td>
  <td>47.65 / 67.09</td>
</tr>
<tr>
  <td>Llama-2-7b</td>
  <td>30.64 / 43.80</td>
  <td>56.64 / 72.14</td>
  <td>46.96 / 66.16</td>
</tr>
<tr>
  <td>Mistral-7B-v0.1</td>
  <td>48.48 / 63.27</td>
  <td>63.54 / 78.73</td>
  <td>53.72 / 72.75</td>
</tr>
<tr>
  <td>SeaLLM-7b-Hybrid</td>
  <td>49.70 / 67.62</td>
  <td>50.62 / 75.21</td>
  <td>49.62 / 70.74</td>
</tr>
<tr>
  <td>SeaLLM-7b-v2</td>
  <td>34.55 / 55.13</td>
  <td>52.21 / 77.00</td>
  <td>46.19 / 72.11</td>
</tr>
<tr>
  <td>Qwen1.5-7B</td>
  <td>53.79 / 69.30</td>
  <td>57.17 / 77.28</td>
  <td>56.63 / 76.99</td>
</tr>
<tr style="color: #000084;">
  <td>Sailor-7B</td>
  <td>57.88 / 71.06</td>
  <td>60.53 / 75.42</td>
  <td>53.81 / 74.62</td>
</tr>
</table>

### Commonsense Reasoning

<table>
  <tr>
    <th>3-shot (EM)</th>
    <th>XCOPA (th)</th>
    <th>XCOPA (id)</th>
    <th>XCOPA (vi)</th>
  </tr>
  <tr style="color: grey">
    <td>Random guess</td>
    <td><span style="color: grey">50.00</span></td>
    <td><span style="color: grey">50.00</span></td>
    <td><span style="color: grey">50.00</span></td>
  </tr>
  <tr>
    <td>Qwen1.5-0.5B</td>
    <td>51.00</td>
    <td>52.20</td>
    <td>53.80</td>
  </tr>
  <tr style="color: #000084;">
    <td>Sailor-0.5B</td>
    <td>51.00</td>
    <td>58.20</td>
    <td>58.00</td>
  </tr>
  <tr>
    <td>Qwen1.5-1.8B</td>
    <td>52.60</td>
    <td>51.60</td>
    <td>53.40</td>
  </tr>
  <tr style="color: #000084;">
    <td>Sailor-1.8B</td>
    <td>53.80</td>
    <td>64.20</td>
    <td>63.20</td>
  </tr>
  <tr>
    <td>Qwen1.5-4B</td>
    <td>53.40</td>
    <td>55.00</td>
    <td>57.80</td>
  </tr>
  <tr style="color: #000084;">
    <td>Sailor-4B</td>
    <td>53.40</td>
    <td>69.20</td>
    <td>68.20</td>
  </tr>
  <tr>
    <td>Llama-2-7b</td>
    <td>52.80</td>
    <td>64.00</td>
    <td>62.00</td>
  </tr>
  <tr>
    <td>Mistral-7B-v0.1</td>
    <td>57.20</td>
    <td>62.40</td>
    <td>61.60</td>
  </tr>
  <tr>
    <td>SeaLLM-7b-Hybrid</td>
    <td>58.20</td>
    <td>71.60</td>
    <td>67.60</td>
  </tr>
  <tr>
    <td>SeaLLM-7b-v2</td>
    <td>56.80</td>
    <td>64.00</td>
    <td>64.60</td>
  </tr>
  <tr>
    <td>Qwen1.5-7B</td>
    <td>54.20</td>
    <td>62.20</td>
    <td>66.20</td>
  </tr>
  <tr style="color: #000084;">
    <td>Sailor-7B</td>
    <td>59.00</td>
    <td>72.20</td>
    <td>72.20</td>
  </tr>
</table>


### Reading Comprehension

<table>
  <tr>
    <th>3-shot (EM)</th>
    <th>Belebele (th)</th>
    <th>Belebele (id)</th>
    <th>Belebele (vi)</th>
  </tr>
  <tr style="color: grey">
    <td>Random guess</td>
    <td><span style="color: grey">25.00</span></td>
    <td><span style="color: grey">25.00</span></td>
    <td><span style="color: grey">25.00</span></td>
  </tr>
  <tr>
    <td>Qwen1.5-0.5B</td>
    <td>29.89</td>
    <td>26.89</td>
    <td>30.22</td>
  </tr>
  <tr style="color: #000084;">
    <td>Sailor-0.5B</td>
    <td>32.22</td>
    <td>30.89</td>
    <td>32.33</td>
  </tr>
  <tr>
    <td>Qwen1.5-1.8B</td>
    <td>30.11</td>
    <td>32.00</td>
    <td>31.33</td>
  </tr>
  <tr style="color: #000084;">
    <td>Sailor-1.8B</td>
    <td>34.22</td>
    <td>34.89</td>
    <td>35.33</td>
  </tr>
  <tr>
    <td>Qwen1.5-4B</td>
    <td>32.78</td>
    <td>36.22</td>
    <td>35.22</td>
  </tr>
  <tr style="color: #000084;">
    <td>Sailor-4B</td>
    <td>36.11</td>
    <td>41.33</td>
    <td>38.89</td>
  </tr>
  <tr>
    <td>Llama-2-7b</td>
    <td>31.78</td>
    <td>39.78</td>
    <td>38.00</td>
  </tr>
  <tr>
    <td>Mistral-7B-v0.1</td>
    <td>34.33</td>
    <td>41.33</td>
    <td>41.33</td>
  </tr>
  <tr>
    <td>SeaLLM-7b-Hybrid</td>
    <td>37.78</td>
    <td>43.11</td>
    <td>43.00</td>
  </tr>
  <tr>
    <td>SeaLLM-7b-v2</td>
    <td>36.33</td>
    <td>43.11</td>
    <td>47.00</td>
  </tr>
  <tr>
    <td>Qwen1.5-7B</td>
    <td>38.33</td>
    <td>42.00</td>
    <td>42.89</td>
  </tr>
  <tr style="color: #000084;">
    <td>Sailor-7B</td>
    <td>41.56</td>
    <td>44.33</td>
    <td>45.33</td>
  </tr>
</table>

### Examination

<table>
  <tr>
    <th>3-shot (EM)</th>
    <th>M3Exam (th)</th>
    <th>M3Exam (id)</th>
    <th>M3Exam (vi)</th>
  </tr>
  <tr style="color: grey">
    <td>Random guess</td>
    <td><span style="color: grey">22.90</span></td>
    <td><span style="color: grey">25.00</span></td>
    <td><span style="color: grey">25.21</span></td>
  </tr>
  <tr>
    <td>Qwen1.5-0.5B</td>
    <td>22.93</td>
    <td>25.07</td>
    <td>26.66</td>
  </tr>
  <tr style="color: #000084;">
    <td>Sailor-0.5B</td>
    <td>24.41</td>
    <td>26.15</td>
    <td>30.91</td>
  </tr>
  <tr>
    <td>Qwen1.5-1.8B</td>
    <td>24.04</td>
    <td>24.26</td>
    <td>28.68</td>
  </tr>
  <tr style="color: #000084;">
    <td>Sailor-1.8B</td>
    <td>25.38</td>
    <td>28.30</td>
    <td>34.71</td>
  </tr>
  <tr>
    <td>Qwen1.5-4B</td>
    <td>24.50</td>
    <td>24.26</td>
    <td>30.02</td>
  </tr>
  <tr style="color: #000084;">
    <td>Sailor-4B</td>
    <td>27.88</td>
    <td>31.27</td>
    <td>40.69</td>
  </tr>
  <tr>
    <td>Llama-2-7b</td>
    <td>23.67</td>
    <td>25.07</td>
    <td>33.15</td>
  </tr>
  <tr>
    <td>Mistral-7B-v0.1</td>
    <td>26.03</td>
    <td>26.68</td>
    <td>36.11</td>
  </tr>
  <tr>
    <td>SeaLLM-7b-Hybrid</td>
    <td>27.18</td>
    <td>26.95</td>
    <td>36.50</td>
  </tr>
  <tr>
    <td>SeaLLM-7b-v2</td>
    <td>28.48</td>
    <td>29.92</td>
    <td>39.18</td>
  </tr>
  <tr>
    <td>Qwen1.5-7B</td>
    <td>25.75</td>
    <td>26.15</td>
    <td>36.28</td>
  </tr>
  <tr style="color: #000084;">
    <td>Sailor-7B</td>
    <td>30.00</td>
    <td>32.88</td>
    <td>44.10</td>
  </tr>
</table>

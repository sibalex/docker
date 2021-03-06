# Docker Machine Learning Build

<img width="1094" alt="Screenshot 2020-08-06 at 7 59 03 PM" src="https://user-images.githubusercontent.com/43387913/89560196-53d0cc80-d81f-11ea-8b21-57462f0871fc.png">

[docs.docker](https://docs.docker.com/engine/reference/commandline/docker/)
[hub.docker](https://hub.docker.com/)
[flasgger.swagger](https://github.com/flasgger/flasgger)
[hub.anaconda3](https://hub.docker.com/r/continuumio/anaconda3)

---

<img width="1325" alt="Screenshot 2020-08-06 at 6 31 12 PM" src="https://user-images.githubusercontent.com/43387913/89552921-7d84f600-d815-11ea-8a6a-3ac79d48599e.png">

> in: 'docker --version'
>
> out: 'Docker version 19.03.12, build 48a66213fe'

> in: 'docker build -t dml_api .'
>
> out: 'Successfully tagged dml_api:latest'

> in: 'docker_ml % docker run -p 8000:8000 dml_api'
>
> out: '* Running on localhost:8000/ (Press CTRL+C to quit)'

---

<img width="694" alt="Screenshot 2020-08-06 at 7 25 19 PM" src="https://user-images.githubusercontent.com/43387913/89558690-326ee100-d81d-11ea-873c-617e15d9c689.png">

> 'pip install streamlit'

> in: 'streamlit run streamlit_app.py'
>
> out: 'You can now view your Streamlit app in your browser.'

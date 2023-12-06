# --------
FROM nbhosting/scipy-notebook:latest

RUN true \
      # the crux of the matter
   && pip install jupyterlab-nbhosting \
      # formerly known as splitcell, used a lot \
   && pip install jupyterlab-gridwidth \
      # used only once to hide one code cell in taylor \
      # formerly known as hide_input \
   && pip install jupyterlab-hidecell \
      # auto-corrected exos \
   && pip install nbautoeval \
      # auto-illustrated examples \
   && pip install ipythontutor \
   && pip install ipywidgets \
      # install plotly for w7
   && pip install plotly \
      # install scikit-image
   && pip install scikit-image \
      # install geopy
   && pip install geopy \
   && true

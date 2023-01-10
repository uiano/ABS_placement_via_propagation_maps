# Select an experiment file:
module_name = "experiments.jpaper_experiments"

# Other settings
# When using mayavi, you may need to comment the line
# "from tvtk.tools.tvtk_doc import TVTKFilterChooser, TVTK_FILTERS"
# at [path_to_your_environment]/lib/python3.6/site-packages/mayavi/filters/user_defined.py
use_mayavi = False
# use_mayavi = True

# GFigure
import gsim.gfigure

gsim.gfigure.title_to_caption = True
gsim.gfigure.default_figsize = (5.5, 3.5)

#log.setLevel(logging.DEBUG)
import logging.config

cfg = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '{levelname}:{name}:{module}: {message}',
            'style': '{',
        },
        'standard': {
            'format': '{levelname}:{asctime}:{name}:{module}: {message}',
            'style': '{',
        },
        'verbose': {
            'format':
            '{levelname}:{asctime}:{name}:{module}:{process:d}:{thread:d}: {message}',
            'style': '{',
        },
    },
    'handlers': {
        # 'file': {
        #     'level': 'INFO',
        #     'class': 'logging.FileHandler',
        #     'filename': os.path.join(BASE_DIR, LOGGING_DIR, 'all.log'),
        #     'formatter': 'standard'
        # },
        'console': {  # This one is overridden in settings_server.py
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
    },
    'loggers': {
        'channel': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': True,
        },
        'placers': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': True,
        },
        'solvers': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': True,
        },
    }
}
logging.config.dictConfig(cfg)
app = FastAPI(
    title='RRF Savant API v5.4',
    description='Production rank fusion engine',
    version='5.4.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

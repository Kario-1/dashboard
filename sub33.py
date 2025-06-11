from fastapi import FastAPI, Request, HTTPException, status, UploadFile, Depends, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Enum, func, JSON, create_engine, or_
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base, joinedload
from sqlalchemy.dialects.postgresql import ARRAY
from dotenv import load_dotenv
from enum import Enum as PyEnum
from pyngrok import ngrok
from huggingface_hub import InferenceClient
from langdetect import detect
import pandas as pd
import io, docx, PyPDF2, json, logging, httpx, hmac, hashlib, uvicorn, os, nest_asyncio

load_dotenv()
Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres.edsmipcwdffklngwxrnd:23OIMyQrZVxzmdea@aws-0-us-east-1.pooler.supabase.com:5432/postgres")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "mnmnhfff")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 3000


client = InferenceClient(
    provider="cohere",
    #api_key="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",
    api_key="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",
)
MODEL_NAME = "CohereLabs/c4ai-command-r7b-12-2024"


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# FastAPI app
app = FastAPI(title="MostasharQ API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only, specify your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Enums
class RoleEnum(str, PyEnum):
    client = "client"
    lawyer = "lawyer"
    company = "company"
    admin = "admin"

class PlanNameEnum(str, PyEnum):
    daily = "daily"
    monthly = "monthly"
    annually = "annually"

class PaymentStatusEnum(str, PyEnum):
    successful = "successful"
    failed = "failed"
    pending = "pending"

class ChatStatusEnum(str, PyEnum):
    active = "active"
    inactive = "inactive"

class ChatTypeEnum(str, PyEnum):
    support = "support"
    lawyer_client = "lawyer_client"
    model_chat = "model_chat"

class GovernorateEnum(str, PyEnum):
    Cairo = "Cairo"
    Alexandria = "Alexandria"
    Giza = "Giza"
    Sharqia = "Sharqia"
    Dakahlia = "Dakahlia"
    Beheira = "Beheira"
    Gharbia = "Gharbia"
    Menoufia = "Menoufia"
    Qalyubia = "Qalyubia"
    PortSaid = "Port Said"
    Suez = "Suez"
    Ismailia = "Ismailia"
    KafrElSheikh = "KafrElSheikh"
    Damietta = "Damietta"
    Assiut = "Assiut"
    Sohag = "Sohag"
    Qena = "Qena"
    Luxor = "Luxor"
    Aswan = "Aswan"
    RedSea = "RedSea"
    NewValley = "NewValley"
    Matrouh = "Matrouh"
    NorthSinai = "NorthSinai"
    SouthSinai = "SouthSinai"
    BeniSuef = "BeniSuef"
    Fayoum = "Fayoum"
    Minya = "Minya"

class WithdrawalStatusEnum(str, PyEnum):
    pending = "pending"
    reviewed = "reviewed"
    paid = "paid"
    rejected = "rejected"

class MessageStatus(str, PyEnum):
    sent = "sent"
    read = "read"

class SupportChatType(str, PyEnum):
    client = "client"
    lawyer = "lawyer"
    company = "company"

class ModelOrderStatus(str, PyEnum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"

class ModelOrderType(str, PyEnum):
    legal_qa = "legal_qa"        
    data_analysis = "data_analysis"  
    custom_training = "custom_training"  

class CardToken(Base):
    __tablename__ = 'card_tokens'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    token = Column(String)
    expiry = Column(DateTime)
    is_default = Column(Boolean, default=True)

# Tables

class DBUser(Base):
    __tablename__ = "User"

    user_id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    email = Column(String(100), unique=True, index=True)
    phone_number = Column(String(20))
    password = Column(String)
    role = Column(Enum(RoleEnum))
    license_number = Column(String(50), nullable=True)
    specialization = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    governorate = Column(Enum(GovernorateEnum), nullable=True)
    lawyer_balance = Column(Float, default=0.0)

    subscriptions = relationship("Subscription", back_populates="user")
    queries = relationship("Queries", back_populates="user")

class Subscription(Base):
    __tablename__ = "subscription"

    sub_id = Column(Integer, primary_key=True, index=True)
    plan_name = Column(Enum(PlanNameEnum))
    price = Column(Float)

    user_id = Column(Integer, ForeignKey("User.user_id"))

    user = relationship("DBUser", back_populates="subscriptions")
    payments = relationship("Payment", back_populates="subscription")
    start_date = Column(DateTime)  
    end_date = Column(DateTime)

    status = Column(String, default='active')
    paymob_subscription_id = Column(String)
    retry_count = Column(Integer, default=0)



class Payment(Base):
    __tablename__ = "payment"

    payment_id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float)
    currency = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)
    payment_status = Column(Enum(PaymentStatusEnum))
    payment_method = Column(String)
    sub_id = Column(Integer, ForeignKey("subscription.sub_id"))

    subscription = relationship("Subscription", back_populates="payments")

class Chat(Base):
    __tablename__ = "chat"

    chat_id = Column(Integer, primary_key=True, index=True)
    status = Column(Enum(ChatStatusEnum))
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    chat_type = Column(Enum(ChatTypeEnum))
    auto_close_at = Column(DateTime, nullable=True)
    participants = Column(ARRAY(Integer), nullable=True)

    queries = relationship("Queries", back_populates="chat")
    responses = relationship("Responses", back_populates="chat")
    logs = relationship("ChatLog", back_populates="chat")
    legal_docs = relationship("LegalDocs", back_populates="chat")

class Queries(Base):
    __tablename__ = "queries"

    query_id = Column(Integer, primary_key=True, index=True)
    query_text = Column(String)
    query_category = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("User.user_id"))
    chat_id = Column(Integer, ForeignKey("chat.chat_id"))
    status = Column(String(20), default="sent")   
    read_at = Column(DateTime, nullable=True)

    user = relationship("DBUser", back_populates="queries")
    chat = relationship("Chat", back_populates="queries")
    responses = relationship("Responses", back_populates="query")

class Responses(Base):
    __tablename__ = "responses"

    responses_id = Column(Integer, primary_key=True, index=True)
    responses_text = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    generated_by = Column(String(50))
    query_id = Column(Integer, ForeignKey("queries.query_id"))
    chat_id = Column(Integer, ForeignKey("chat.chat_id"))
    status = Column(String(20), default="sent", nullable=False)
    read_at = Column(DateTime, nullable=True)

    query = relationship("Queries", back_populates="responses")
    chat = relationship("Chat", back_populates="responses")

class ChatLog(Base):
    __tablename__ = "chat_log"

    log_id = Column(Integer, primary_key=True, index=True)
    session_startat = Column(DateTime)
    session_endat = Column(DateTime)
    chat_id = Column(Integer, ForeignKey("chat.chat_id"))

    chat = relationship("Chat", back_populates="logs")

class LegalDocs(Base):
    __tablename__ = "legal_docs"

    doc_id = Column(Integer, primary_key=True, index=True)
    doc_name = Column(String(100))
    doc_type = Column(String(50))
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    chat_id = Column(Integer, ForeignKey("chat.chat_id"))

    chat = relationship("Chat", back_populates="legal_docs")

class MessageQuota(Base):
    __tablename__ = "message_quota"

    quota_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("User.user_id"))
    remaining_messages = Column(Integer)
    total_messages = Column(Integer)
    expiry_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("DBUser", backref="message_quota")

class SupportChat(Base):
    __tablename__ = "support_chat"

    chat_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("User.user_id"))
    status = Column(Enum(ChatStatusEnum))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    auto_close_at = Column(DateTime)  

    user = relationship("DBUser", backref="support_chats")
    messages = relationship("SupportMessage", back_populates="chat")

class SupportMessage(Base):
    __tablename__ = "support_message"

    message_id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("support_chat.chat_id"))
    sender_id = Column(Integer, ForeignKey("User.user_id"))
    content = Column(String)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    chat = relationship("SupportChat", back_populates="messages")
    sender = relationship("DBUser", backref="support_messages")

class ChatSummary(Base):
    __tablename__ = "chat_summary"

    summary_id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("queries.query_id"))
    response_id = Column(Integer, ForeignKey("responses.responses_id"))
    question_summary = Column(String)
    answer_summary = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    query = relationship("Queries")
    response = relationship("Responses")

# ==================================================
# Pydantic Models for Request/Response
# ==================================================

# Chat-related models
class MChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 8192

class MChatResponse(BaseModel):
    response: str
    chat_id: str

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime

class FullChatResponse(BaseModel):
    chat_id: int
    status: ChatStatusEnum
    started_at: datetime
    ended_at: Optional[datetime]
    messages: List[ChatMessage]
    participants: List[int]
    can_reply: bool = False

class LChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime

class StartChatResponse(BaseModel):
    chat_id: int
    lawyer_name: str
    messages: List[LChatMessage] = []

class ChatWithLawyerRequest(BaseModel):
    lawyer_id: int

class SuccessResponse(BaseModel):
    message: str

class LawyerChatSummary(BaseModel):
    chat_id: int
    participant_id: int
    participant_name: str
    last_message: Optional[str]
    last_message_time: Optional[datetime]
    unread_count: int
    status: ChatStatusEnum
    created_at: datetime

    class Config:
        from_attributes = True

# Admin models
class AdminModelQuery(BaseModel):
    question: str
    chat_id: int
    temperature: float = 0.7
    max_tokens: int = 1000
    selected_sources: Optional[List[str]] = None 


class AdminModelResponse(BaseModel):
    response: str
    processing_time: float
    tokens_used: int
    confidence_score: float
    data_sources: List[str]

# Model interaction models
class ModelQuery(BaseModel):
    question: str = Form(...)
    chat_id: int = Form(...)
    temperature: float = Form(...)
    max_tokens: int = Form(...)
    file_name: str = Form(...)
    use_uploaded_file: bool = Form(...)

class ModelResponse(BaseModel):
    response: str
    processing_time: float
    tokens_used: int
    confidence_score: float
    chat_id: int
    query_id: int

class ModelChatHistory(BaseModel):
    chat_id: int
    messages: List[dict]
    status: ChatStatusEnum
    auto_close_at: Optional[datetime]

# Lawyer models
class LawyerResponse(BaseModel):
    user_id: int
    first_name: str
    last_name: str
    email: str
    phone_number: str
    specialization: str
    governorate: GovernorateEnum
    lawyer_balance: float

    class Config:
        from_attributes = True

# Document processing models
class DocumentSummaryRequest(BaseModel):
    chat_id: int
    temperature: float = 0.7
    max_tokens: int = 1000

class DocumentSummaryResponse(BaseModel):
    summary: str
    processing_time: float
    tokens_used: int
    confidence_score: float

class CaseAnalysisRequest(BaseModel):
    chat_id: int
    temperature: float = 0.7
    max_tokens: int = 1000

class CaseAnalysisResponse(BaseModel):
    analysis: str
    processing_time: float
    tokens_used: int
    confidence_score: float

# User models
class UserBase(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    phone_number: Optional[str] = None

class UserCreate(UserBase):
    password: str
    role: RoleEnum
    license_number: Optional[str] = None
    governorate: GovernorateEnum 

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    password: Optional[str] = None
    license_number: Optional[str] = None  
    specialization: Optional[str] = None  

    class Config:
        from_attributes = True



class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class UserResponse(UserBase):
    user_id: int
    role: RoleEnum
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int

class TokenData(BaseModel):
    email: Optional[str] = None

class SubscriptionBase(BaseModel):
    plan_name: PlanNameEnum

class SubscriptionCreate(SubscriptionBase):
    pass

class SubscriptionResponse(SubscriptionBase):
    sub_id: int
    start_date: datetime  
    end_date: datetime   
    status: str         
    user_id: int
    price: float

    class Config:
        from_attributes = True

class SubscriptionUpdate(BaseModel):
    plan_name: Optional[PlanNameEnum] = None
    status: Optional[str] = None  

class PaymentBase(BaseModel):
    amount: float
    currency: str
    payment_status: PaymentStatusEnum
    payment_method: str

class PaymentCreate(PaymentBase):
    pass

class PaymentResponse(PaymentBase):
    id: int
    subscription_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Payment models
class PaymentRequest(BaseModel):
    plan_name: PlanNameEnum
    user_id: int
    class Config:
        arbitrary_types_allowed = True

class PaymentResponse(BaseModel):
    payment_url: str

class ChatBase(BaseModel):
    status: ChatStatusEnum

class ChatCreate(ChatBase):
    pass

class ChatResponse(ChatBase):
    id: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    user_id: int

    class Config:
        from_attributes = True

class ChatLogBase(BaseModel):
    session_start: datetime
    session_end: datetime
    summary: Optional[str] = None

class ChatLogCreate(ChatLogBase):
    pass

class ChatLogResponse(ChatLogBase):
    id: int
    chat_id: int

    class Config:
        from_attributes = True

class LegalDocBase(BaseModel):
    doc_name: str
    doc_type: str

class LegalDocCreate(LegalDocBase):
    pass

class LegalDocResponse(LegalDocBase):
    id: int
    chat_id: int
    uploaded_at: datetime
    file_path: str

    class Config:
        from_attributes = True

class QueryBase(BaseModel):
    query_text: str

class QueryCreate(QueryBase):
    pass

class QueryResponse(QueryBase):
    id: int
    created_at: datetime
    user_id: int
    chat_id: int

    class Config:
        from_attributes = True

class ResponseBase(BaseModel):
    response_text: str
    generated_by: str

class ResponseCreate(ResponseBase):
    pass

class ResponseResponse(ResponseBase):
    id: int
    created_at: datetime
    query_id: int
    chat_id: int

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    chat_id: int
    status: ChatStatusEnum
    started_at: datetime
    ended_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    chat_id: int
    title: str
    status: ChatStatusEnum
    started_at: datetime
    ended_at: Optional[datetime] = None
    last_message_preview: Optional[str] = None

    class Config:
        from_attributes = True



class MessageQuotaResponse(BaseModel):
    remaining_messages: int
    total_messages: int
    expiry_date: datetime

    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    total_accounts: int
    total_users: int
    total_lawyers: int
    total_companies: int
    total_active_subscriptions: int
    total_revenue: float
    monthly_revenue: float
    active_chats: int
    total_messages: int

    class Config:
        from_attributes = True

class UserStats(BaseModel):
    user_id: int
    email: str
    role: RoleEnum
    subscription_status: Optional[str]
    subscription_end_date: Optional[datetime]
    total_messages: int
    remaining_messages: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True

class RevenueStats(BaseModel):
    date: datetime
    amount: float
    subscription_type: PlanNameEnum
    user_id: int

    class Config:
        from_attributes = True


class SupportChat(BaseModel):
    chat_id: int
    user_id: int
    user_name: str
    user_email: str
    user_type: SupportChatType
    last_message: Optional[str]
    last_message_time: Optional[datetime]
    unread_count: int
    status: ChatStatusEnum
    created_at: datetime

    class Config:
        from_attributes = True

class SupportMessage(BaseModel):
    message_id: int
    chat_id: int
    sender_id: int
    sender_name: str
    sender_type: str
    content: str
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True


class WithdrawalRequest(Base):
    __tablename__ = "withdrawal_requests"

    request_id = Column(Integer, primary_key=True, index=True)
    lawyer_id = Column(Integer, ForeignKey("User.user_id"))
    amount = Column(Float)
    phone_number = Column(String(20))
    status = Column(Enum(WithdrawalStatusEnum), default=WithdrawalStatusEnum.pending)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    admin_notes = Column(String, nullable=True)

    lawyer = relationship("DBUser", backref="withdrawal_requests")

class WithdrawalRequestCreate(BaseModel):
    amount: float
    phone_number: str
class WithdrawalRequestResponse(BaseModel):
    request_id: int
    lawyer_id: int
    lawyer_name: str
    amount: float
    phone_number: str
    status: WithdrawalStatusEnum
    created_at: datetime
    updated_at: datetime
    admin_notes: Optional[str] = None

    class Config:
        from_attributes = True

class WithdrawalStatusUpdate(BaseModel):
    status: WithdrawalStatusEnum
    admin_notes: Optional[str] = None

class ModelOrder(BaseModel):
    order_id: int
    company_id: int
    order_type: ModelOrderType
    description: str
    database_schema: Optional[dict] 
    training_data: Optional[dict]    
    status: ModelOrderStatus
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    admin_notes: Optional[str]
    error_message: Optional[str]

class ModelOrderCreate(BaseModel):
    order_type: ModelOrderType
    description: str
    database_schema: Optional[dict]
    training_data: Optional[dict]

class ModelOrderResponse(BaseModel):
    order_id: int
    company_id: int
    order_type: ModelOrderType
    description: str
    status: ModelOrderStatus
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    admin_notes: Optional[str]

class ModelOrderDB(Base):
    __tablename__ = "model_orders"

    order_id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("User.user_id"))
    order_type = Column(Enum(ModelOrderType))
    description = Column(String)
    database_schema = Column(JSON, nullable=True)
    training_data = Column(JSON, nullable=True)
    status = Column(Enum(ModelOrderStatus), default=ModelOrderStatus.pending)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    admin_notes = Column(String, nullable=True)
    error_message = Column(String, nullable=True)

    company = relationship("DBUser", backref="model_orders")

# Dictionary to store user chats
user_chats: Dict[str, List[MChatMessage]] = {}

# ==================================================
# Database & Utility Functions
# ==================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_by_email(db: Session, email: str):
    return db.query(DBUser).filter(DBUser.email == email).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = pwd_context.hash(user.password)
    db_user = DBUser(
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        phone_number=user.phone_number,
        password=hashed_password,
        role=user.role,
        license_number=user.license_number,
        governorate=user.governorate
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return False
    if not pwd_context.verify(password, user.password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception

    user = get_user_by_email(db, token_data.email)
    if user is None:
        raise credentials_exception
    return user

def check_active_subscription(db: Session, user_id: int):
    return db.query(Subscription).filter(
        Subscription.user_id == user_id,
        Subscription.end_date >= datetime.utcnow(),
        Subscription.status == "active"
    ).first()

def get_plan_details(plan_name: PlanNameEnum):
    """
    Get plan details

    Args:
        plan_name (PlanNameEnum): Requested plan type

    Returns:
        dict: Plan details

    Raises:
        HTTPException: If plan type is invalid
    """
    plans = {
        PlanNameEnum.daily: {"price": 200, "duration_days": 1, "message_quota": 50},
        PlanNameEnum.monthly: {"price": 2000, "duration_days": 30, "message_quota": 500},
        PlanNameEnum.annually: {"price": 20000, "duration_days": 365, "message_quota": 6000}
    }

    plan_details = plans.get(plan_name)
    if not plan_details:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plan: {plan_name}. Available plans: {', '.join([p.value for p in PlanNameEnum])}"
        )

    return plan_details


def detect_language(text: str) -> str:
    """Detect the language of the input text"""
    try:
        return detect(text)
    except:
        return "ar"  # Default to English if detection fails

def load_legal_files():
    """Load all legal files from the data directory"""
    legal_data = {}
    data_dir = "rag_documents"
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                    try:
                        legal_data[filename] = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"Error loading {filename}: {str(e)}")

    return legal_data

def identify_relevant_file(query: str, legal_data: Dict) -> str:
    """Identify which legal file is most relevant to the query"""
    system_prompt = f"""
    You are a legal expert. Analyze the following question and identify which legal file it relates to.
    Available files: {list(legal_data.keys())}
    Question: {query}
    Return only the filename that is most relevant.
    Do not explain your answer. Do not return anything else. Just one filename, exactly as-is,  No extra text, no punctuation, no explanation.
    """
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": system_prompt}]
    )
    print(completion.choices[0].message["content"].strip())
    return completion.choices[0].message["content"].strip()

def format_chat_history(chat_history: List[MChatMessage]) -> str:
    """Format chat history"""
    formatted = ""
    for msg in chat_history:
        formatted += f"{msg.role}: {msg.content}\n"
    return formatted

# ==================================================
# Authentication APIs
# ==================================================

@app.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    try:
        # Validate email format
        if not user.email or '@' not in user.email:
            raise HTTPException(
                status_code=400,
                detail="Invalid email format"
            )

        # Check if email already exists
        db_user = get_user_by_email(db, user.email)
        if db_user:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )

        # Validate password strength
        if len(user.password) < 8:
            raise HTTPException(
                status_code=400,
                detail="Password must be at least 8 characters long"
            )

        # Validate role-specific fields
        if user.role == RoleEnum.lawyer and not user.license_number:
            raise HTTPException(
                status_code=400,
                detail="License number is required for lawyers"
            )

        return create_user(db, user)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during registration: {str(e)}"
        )

@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        if not form_data.username or not form_data.password:
            raise HTTPException(
                status_code=400,
                detail="Email and password are required"
            )

        user = authenticate_user(db, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer", "user_id":user.user_id }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during login: {str(e)}"
        )

@app.post("/logout")
def logout(token: str = Depends(oauth2_scheme)):
    return {"message": "Successfully logged out"}

# ==================================================
# User Management APIs
# ==================================================

@app.get("/users/me", response_model=UserResponse)
def get_user(
    user_id: int,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    """Get user information"""
    check_permissions(current_user, owner_id=user_id)

    user = db.query(DBUser).filter(DBUser.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/users/me", response_model=UserResponse)
async def update_current_user(
    update_data: UserUpdate,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update the currently authenticated user's details.

    - Only updates provided fields (partial update supported)
    - Password will be hashed automatically if provided
    - Lawyers can update license_number and specialization
    """

    update_values = update_data.dict(exclude_unset=True)

    # Handle password update
    if 'password' in update_values:
        update_values['password'] = pwd_context.hash(update_values['password'])

    # Validate lawyer-specific fields
    if current_user.role != RoleEnum.lawyer or current_user.role != RoleEnum.admin:
        if 'license_number' in update_values or 'specialization' in update_values:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only lawyers can update license number or specialization"
            )

    # Update only the fields that were provided
    for field, value in update_values.items():
        setattr(current_user, field, value)

    db.add(current_user)
    db.commit()
    db.refresh(current_user)

    return current_user

@app.post("/users/me/change-password", response_model=dict)
def change_password(
    password_data: ChangePasswordRequest,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Change user's password after verifying old password.

    Steps:
    1. Verify old password matches current password
    2. Hash new password
    3. Update user record
    """

    # 1. Verify old password
    if not pwd_context.verify(password_data.old_password, current_user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Old password is incorrect"
        )

    # 2. Hash new password
    new_hashed_password = pwd_context.hash(password_data.new_password)

    # 3. Update user record
    current_user.password = new_hashed_password
    db.add(current_user)
    db.commit()

    return {"message": "Password changed successfully"}

@app.get("/users/me/chat-history", response_model=List[ChatHistoryResponse])
def get_user_chat_history(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 20,
    offset: int = 0
):
    chats = db.query(Chat).join(Queries).filter(
        Queries.user_id == current_user.user_id
    ).distinct().order_by(
        Chat.started_at.desc()
    ).offset(offset).limit(limit).all()

    history = []
    for chat in chats:
        first_real_query = db.query(Queries).filter(
            Queries.chat_id == chat.chat_id,
            Queries.user_id == current_user.user_id
        ).order_by(Queries.created_at.asc()).first()

        chat_title = first_real_query.query_text if first_real_query else "New chat"

        history.append({
            "chat_id": chat.chat_id,
            "title": chat_title,
            "status": chat.status,
            "started_at": chat.started_at,
            "ended_at": chat.ended_at
        })

    return history

@app.get("/users/me/message-quota", response_model=MessageQuotaResponse)
def get_user_message_quota(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's remaining message quota"""
    quota = db.query(MessageQuota).filter(
        MessageQuota.user_id == current_user.user_id,
        MessageQuota.expiry_date > datetime.utcnow()
    ).first()

    if not quota:
        raise HTTPException(
            status_code=404,
            detail="No active message quota found"
        )

    return quota




# ==================================================
# Subscription Management APIs
# ==================================================


@app.get("/subscriptions", response_model=List[SubscriptionResponse])
def get_all_subscriptions(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Allow all users to view THEIR OWN subscriptions
    subscriptions = db.query(Subscription).filter(
        Subscription.user_id == current_user.user_id
    ).all()
    return subscriptions

@app.post("/subscriptions", response_model=SubscriptionResponse)
def create_subscription(
    subscription: SubscriptionCreate,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Check for existing active subscription
        if check_active_subscription(db, current_user.user_id):
            raise HTTPException(
                status_code=400,
                detail="User already has an active subscription"
            )

        # Validate subscription plan
        if subscription.plan_name not in PlanNameEnum:
            raise HTTPException(
                status_code=400,
                detail="Invalid subscription plan"
            )

        # Get automatic plan details
        plan_details = get_plan_details(subscription.plan_name)

        # Calculate dates
        start_date = datetime.utcnow()
        end_date = start_date + timedelta(days=plan_details["duration_days"])

        # Create subscription
        db_subscription = Subscription(
            plan_name=subscription.plan_name,
            price=plan_details["price"],
            start_date=start_date,
            end_date=end_date,
            status="active",
            user_id=current_user.user_id
        )

        db.add(db_subscription)
        db.flush()

        # Create message quota
        message_quota = MessageQuota(
            user_id=current_user.user_id,
            remaining_messages=plan_details["message_quota"],
            total_messages=plan_details["message_quota"],
            expiry_date=end_date
        )

        db.add(message_quota)
        db.commit()
        db.refresh(db_subscription)

        return db_subscription
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while creating subscription: {str(e)}"
        )


# ==================================================
# Chat Management APIs
# ==================================================
@app.post("/chats", response_model=ChatResponse)
def create_chat(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        active_chats = db.query(Chat).join(Queries).filter(
            Queries.user_id == current_user.user_id,
            Chat.status == ChatStatusEnum.active
        ).count()

        if active_chats >= 5:
            raise HTTPException(
                status_code=400,
                detail="You have reached the maximum number of active chats"
            )

        auto_close_at = datetime.now(timezone.utc) + timedelta(days=1)

        db_chat = Chat(
            status=ChatStatusEnum.active,
            started_at=datetime.now(timezone.utc),
            chat_type=ChatTypeEnum.model_chat,
            auto_close_at=auto_close_at,
            participants=[current_user.user_id, 1]
        )
        db.add(db_chat)
        db.commit()
        db.refresh(db_chat)

        return db_chat

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while creating the chat: {str(e)}"
        )


@app.put("/chats/{chat_id}/end", response_model=ChatResponse)
def end_chat(
    chat_id: int,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    End an active chat session.
    """
    chat = db.query(Chat).options(
        joinedload(Chat.queries)
    ).filter(Chat.chat_id == chat_id).first()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.status == ChatStatusEnum.inactive:
        raise HTTPException(
            status_code=400,
            detail="Chat is already ended"
        )
    
    is_participant = current_user.user_id in chat.participants

    is_admin = current_user.role == RoleEnum.admin

    if not is_participant and not is_admin:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to end this chat"
        )

    try:
        chat.status = ChatStatusEnum.inactive
        chat.ended_at = datetime.utcnow()
        db.commit()
        db.refresh(chat)
        return chat
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to end chat: {str(e)}"
        )




@app.get("/chats/{chat_id}/open", response_model=FullChatResponse)
async def open_chat(
    chat_id: int,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = db.query(Chat).filter(Chat.chat_id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="The conversation does not exist.")

    is_participant = db.query(Queries).filter(
        Queries.chat_id == chat_id,
        Queries.user_id == current_user.user_id
    ).first() is not None

    if not is_participant and current_user.role != RoleEnum.admin:
        raise HTTPException(status_code=403, detail="You are not authorized to access this conversation.")

    messages = []
    queries = db.query(Queries).filter(
        Queries.chat_id == chat_id
    ).order_by(Queries.created_at).all()

    for query in queries:
        messages.append(ChatMessage(
            role="user",
            content=query.query_text,
            timestamp=query.created_at
        ))

    responses = db.query(Responses).filter(
        Responses.chat_id == chat_id
    ).order_by(Responses.created_at).all()

    for response in responses:
        messages.append(ChatMessage(
            role="assistant" if response.generated_by == "model" else "lawyer",
            content=response.responses_text,
            timestamp=response.created_at
        ))

    participants = {current_user.user_id}
    if chat.chat_type == ChatTypeEnum.lawyer_client:
        lawyer_response = db.query(Responses).filter(
            Responses.chat_id == chat_id,
            Responses.generated_by.like("lawyer_%")
        ).first()
        if lawyer_response:
            lawyer_id = int(lawyer_response.generated_by.split("_")[1])
            participants.add(lawyer_id)

    return FullChatResponse(
        chat_id=chat.chat_id,
        status=chat.status,
        started_at=chat.started_at,
        ended_at=chat.ended_at,
        messages=messages,
        participants=list(participants),
        can_reply=chat.status == ChatStatusEnum.active
    )


# =================================================
# Lawyer Chat APIs
# ==================================================




@app.post("/chats/with-lawyer", response_model=StartChatResponse)
def start_chat_with_lawyer(
    chat_request: ChatWithLawyerRequest,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        lawyer = db.query(DBUser).filter(
            DBUser.user_id == chat_request.lawyer_id,
            DBUser.role == RoleEnum.lawyer
        ).first()

        if not lawyer:
            raise HTTPException(
                status_code=404,
                detail="Lawyer not found or not active"
            )

        existing_chat = db.query(Chat).filter(
            Chat.chat_type == ChatTypeEnum.lawyer_client,
            Chat.status == ChatStatusEnum.active,
            Chat.participants.contains([current_user.user_id]),
            Chat.participants.contains([chat_request.lawyer_id])
        ).first()

        chat_to_return = None
        messages_to_return = []

        if existing_chat:
            chat_to_return = existing_chat
            all_chat_entries = []
            queries_db = db.query(Queries).filter(Queries.chat_id == chat_to_return.chat_id).all()
            for query in queries_db:
                all_chat_entries.append({"type": "query", "data": query})

            responses_db = db.query(Responses).filter(Responses.chat_id == chat_to_return.chat_id).all()
            for response in responses_db:
                all_chat_entries.append({"type": "response", "data": response})

            all_chat_entries.sort(key=lambda x: x["data"].created_at)

            for entry in all_chat_entries:
                if entry["type"] == "query":
                    query = entry["data"]
                    messages_to_return.append(LChatMessage(
                        role="user",
                        content=query.query_text,
                        timestamp=query.created_at
                    ))
                elif entry["type"] == "response":
                    response = entry["data"]
                    role = "assistant"
                    if response.generated_by.startswith("lawyer_"):
                        role = "lawyer"
                    messages_to_return.append(LChatMessage(
                        role=role,
                        content=response.responses_text,
                        timestamp=response.created_at
                    ))

            if current_user.role == RoleEnum.lawyer:
                db.query(Queries).filter(
                    Queries.chat_id == chat_to_return.chat_id,
                    Queries.status == "sent",
                    Queries.user_id != current_user.user_id
                ).update({"status": "read", "read_at": datetime.utcnow()})
            else:
                db.query(Responses).filter(
                    Responses.chat_id == chat_to_return.chat_id,
                    Responses.status == "sent",
                    or_(
                        Responses.generated_by.like("lawyer_%"),
                        Responses.generated_by == "model"
                    )
                ).update({"status": "read", "read_at": datetime.utcnow()})
            db.commit()

        else:
            quota = db.query(MessageQuota).filter(
                MessageQuota.user_id == current_user.user_id,
                MessageQuota.expiry_date > datetime.utcnow(),
                MessageQuota.remaining_messages >= 5
            ).first()

            if not quota:
                raise HTTPException(
                    status_code=403,
                    detail="No remaining messages. Please purchase more messages to continue."
                )

            quota.remaining_messages -= 5
            db.add(quota)

            lawyer.lawyer_balance += 5.0
            db.add(lawyer)

            new_chat = Chat(
                status=ChatStatusEnum.active,
                started_at=datetime.utcnow(),
                chat_type=ChatTypeEnum.lawyer_client,
                participants=[current_user.user_id, chat_request.lawyer_id]
            )
            db.add(new_chat)
            db.flush()
            db.commit()
            chat_to_return = new_chat
            messages_to_return = []

        return StartChatResponse(
            chat_id=chat_to_return.chat_id,
            lawyer_name=f"{lawyer.first_name} {lawyer.last_name}",
            messages=messages_to_return
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while starting lawyer chat: {str(e)}"
        )


@app.post("/chats/{chat_id}/messages", response_model=SuccessResponse)
def send_message(
    chat_id: int,
    message: str,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message in a chat"""
    quota = db.query(MessageQuota).filter(
        MessageQuota.user_id == current_user.user_id,
        MessageQuota.expiry_date > datetime.utcnow(),
        MessageQuota.remaining_messages > 0
    ).first()

    if not quota:
        raise HTTPException(
            status_code=403,
            detail="No remaining messages in quota"
        )

    chat = db.query(Chat).filter(
        Chat.chat_id == chat_id,
        Chat.status == ChatStatusEnum.active
    ).first()

    if not chat:
        raise HTTPException(status_code=404, detail="Active chat not found")

    new_message = Queries(
        query_text=message,
        user_id=current_user.user_id,
        chat_id=chat_id,
        created_at=datetime.utcnow()
    )
    db.add(new_message)

    quota.remaining_messages -= 1
    db.add(quota)

    db.commit()
    db.refresh(new_message)
    return SuccessResponse(message="Message sent successfully.")

@app.post("/chats/{chat_id}/lawyer-response", response_model=SuccessResponse)
def add_lawyer_response(
    chat_id: int,
    response: str,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add a lawyer's response to a chat.
    """
    if current_user.role != RoleEnum.lawyer:
        raise HTTPException(
            status_code=403,
            detail="Only lawyers can add responses"
        )

    chat = db.query(Chat).filter(
        Chat.chat_id == chat_id,
        Chat.status == ChatStatusEnum.active
    ).first()

    if not chat:
        raise HTTPException(status_code=404, detail="Active chat not found")

    last_query = db.query(Queries).filter(
        Queries.chat_id == chat_id
    ).order_by(Queries.created_at.desc()).first()

    if not last_query:
        raise HTTPException(
            status_code=400,
            detail="No queries found in this chat"
        )

    new_response = Responses(
        responses_text=response,
        generated_by=f"lawyer_{current_user.user_id}",
        query_id=last_query.query_id,
        chat_id=chat_id,
        created_at=datetime.utcnow()
    )
    db.add(new_response)

    current_user.lawyer_balance += 1.0


    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return SuccessResponse(message="Lawyer response successfully added.")

@app.get("/chats/{chat_id}/full", response_model=FullChatResponse)
async def get_full_chat_history(
    chat_id: int,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = db.query(Chat).filter(Chat.chat_id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    is_participant = False
    if current_user.user_id in chat.participants:
        is_participant = True

    if not is_participant and current_user.role != RoleEnum.admin:
        raise HTTPException(status_code=403, detail="You are not authorized to access this chat")

    if current_user.role == RoleEnum.lawyer:
        db.query(Queries).filter(
            Queries.chat_id == chat_id,
            Queries.status == "sent",
            Queries.user_id != current_user.user_id
        ).update({"status": "read", "read_at": datetime.utcnow()})
    else:
        db.query(Responses).filter(
            Responses.chat_id == chat_id,
            Responses.status == "sent",
            or_(
                Responses.generated_by.like("lawyer_%"),
                Responses.generated_by == "model"
            )
            ).update({"status": "read", "read_at": datetime.utcnow()})

    db.commit()

    all_chat_entries = []
    participants = set()

    queries_db = db.query(Queries).filter(Queries.chat_id == chat_id).all()
    for query in queries_db:
        all_chat_entries.append({
            "type": "query",
            "data": query
        })
        participants.add(query.user_id)

    responses_db = db.query(Responses).filter(Responses.chat_id == chat_id).all()
    for response in responses_db:
        all_chat_entries.append({
            "type": "response",
            "data": response
        })
        if response.generated_by.startswith("lawyer_"):
            try:
                lawyer_id = int(response.generated_by.split("_")[1])
                participants.add(lawyer_id)
            except ValueError:
                pass

    all_chat_entries.sort(key=lambda x: x["data"].created_at)

    messages = []
    for entry in all_chat_entries:
        if entry["type"] == "query":
            query = entry["data"]
            messages.append(ChatMessage(
                role="user",
                content=query.query_text,
                timestamp=query.created_at
            ))
        elif entry["type"] == "response":
            response = entry["data"]
            role = "assistant"
            if response.generated_by.startswith("lawyer_"):
                role = "lawyer"
            messages.append(ChatMessage(
                role=role,
                content=response.responses_text,
                timestamp=response.created_at
            ))

    return FullChatResponse(
        chat_id=chat.chat_id,
        status=chat.status,
        started_at=chat.started_at,
        ended_at=chat.ended_at,
        messages=messages,
        participants=list(participants)
    )


# ==================================================
# Admin Dashboard APIs
# ==================================================

def check_admin_access(current_user: DBUser):
    """Helper function to check if user is admin"""
    if current_user.role != RoleEnum.admin:
        raise HTTPException(
            status_code=403,
            detail="Only admin can access this endpoint"
        )
    return True

def check_permissions(current_user: DBUser, required_role: Optional[RoleEnum] = None, owner_id: Optional[int] = None):
    if current_user.role == RoleEnum.admin:
        return True

    if required_role and current_user.role != required_role:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to perform this action"
        )

    if owner_id and current_user.user_id != owner_id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this resource"
        )

    return True

@app.get("/admin/dashboard/stats", response_model=DashboardStats)
def get_dashboard_stats(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get overall dashboard statistics"""
    check_admin_access(current_user)

    # Get total users by role
    total_accounts = db.query(DBUser).count()
    total_users = db.query(DBUser).filter(DBUser.role == RoleEnum.client).count()
    total_lawyers = db.query(DBUser).filter(DBUser.role == RoleEnum.lawyer).count()
    total_companies = db.query(DBUser).filter(DBUser.role == RoleEnum.company).count()

    # Get subscription stats
    active_subscriptions = db.query(Subscription).filter(
        Subscription.status == "active",
        Subscription.end_date > datetime.utcnow()
    ).count()

    # Get revenue stats
    total_revenue = db.query(func.sum(Payment.amount)).filter(
        Payment.payment_status == PaymentStatusEnum.successful
    ).scalar() or 0.0

    # Get monthly revenue
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    monthly_revenue = db.query(func.sum(Payment.amount)).filter(
        Payment.payment_status == PaymentStatusEnum.successful,
        Payment.created_at >= month_start
    ).scalar() or 0.0

    # Get chat stats
    active_chats = db.query(Chat).filter(
        Chat.status == ChatStatusEnum.active
    ).count()

    total_messages = db.query(Queries).count()

    return DashboardStats(
        total_accounts=total_accounts,
        total_users=total_users,
        total_lawyers=total_lawyers,
        total_companies=total_companies,
        total_active_subscriptions=active_subscriptions,
        total_revenue=total_revenue,
        monthly_revenue=monthly_revenue,
        active_chats=active_chats,
        total_messages=total_messages
    )

@app.get("/admin/users", response_model=List[UserStats])
def get_all_users(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """Get all users with their stats"""
    check_admin_access(current_user)

    users = db.query(DBUser).offset(skip).limit(limit).all()
    user_stats = []

    for user in users:
        # Get subscription info
        subscription = db.query(Subscription).filter(
            Subscription.user_id == user.user_id,
            Subscription.status == "active",
            Subscription.end_date > datetime.utcnow()
        ).first()

        # Get message quota
        quota = db.query(MessageQuota).filter(
            MessageQuota.user_id == user.user_id,
            MessageQuota.expiry_date > datetime.utcnow()
        ).first()

        # Get total messages
        total_messages = db.query(Queries).filter(
            Queries.user_id == user.user_id
        ).count()

        user_stats.append(UserStats(
            user_id=user.user_id,
            email=user.email,
            role=user.role,
            subscription_status=subscription.status if subscription else None,
            subscription_end_date=subscription.end_date if subscription else None,
            total_messages=total_messages,
            remaining_messages=quota.remaining_messages if quota else None,
            created_at=user.created_at
        ))

    return user_stats


@app.get("/admin/{user_id}/subscription", response_model=SubscriptionResponse)
def get_user_subscription(
    user_id: int,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        user = db.query(DBUser).filter(DBUser.user_id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        if current_user.role != RoleEnum.admin :
            raise HTTPException(
                status_code=403,
                detail="Not authorized to access this subscription"
            )

        subscription = db.query(Subscription).filter(
            Subscription.user_id == user_id,
            Subscription.end_date >= datetime.utcnow()
        ).order_by(Subscription.end_date.desc()).first()

        if not subscription:
            raise HTTPException(
                status_code=404,
                detail="No active subscription found"
            )

        return subscription
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/admin/revenue", response_model=List[RevenueStats])
def get_revenue_stats(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Get revenue statistics"""
    check_admin_access(current_user)

    query = db.query(
        Payment.created_at,
        Payment.amount,
        Subscription.plan_name,
        Payment.sub_id
    ).join(
        Subscription,
        Payment.sub_id == Subscription.sub_id
    ).filter(
        Payment.payment_status == PaymentStatusEnum.successful
    )

    if start_date:
        query = query.filter(Payment.created_at >= start_date)
    if end_date:
        query = query.filter(Payment.created_at <= end_date)

    payments = query.order_by(Payment.created_at.desc()).all()

    return [
        RevenueStats(
            date=payment.created_at,
            amount=payment.amount,
            subscription_type=payment.plan_name,
            user_id=payment.sub_id
        )
        for payment in payments
    ]

@app.get("/admin/subscriptions", response_model=List[SubscriptionResponse])
def get_all_subscriptions(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
):
    """Get all subscriptions (admin only)"""
    check_admin_access(current_user)

    query = db.query(Subscription)
    if status:
        query = query.filter(Subscription.status == status)

    subscriptions = query.offset(skip).limit(limit).all()
    return subscriptions

@app.get("/admin/model/orders", response_model=List[ModelOrderResponse])
async def get_all_model_orders(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    status: Optional[ModelOrderStatus] = None,
    company_id: Optional[int] = None
):
    """Get all model orders (admin only)"""
    if current_user.role != RoleEnum.admin:
        raise HTTPException(status_code=403, detail="Not authorized to access this data")

    query = db.query(ModelOrderDB)

    if status:
        query = query.filter(ModelOrderDB.status == status)

    if company_id:
        query = query.filter(ModelOrderDB.company_id == company_id)

    orders = query.order_by(ModelOrderDB.created_at.desc()).all()
    return orders

# ==================================================
# Admin Model Chat APIs
# ==================================================

@app.post("/admin/model/chat", response_model=AdminModelResponse)
async def admin_model_chat(
    query: AdminModelQuery,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Chat with the model as admin with full database access, with selectable data sources."""
    try:
        if current_user.role != RoleEnum.admin:
            raise HTTPException(
                status_code=403,
                detail="Only admin can access this endpoint"
            )

        start_time = datetime.utcnow()

        # هنا هتبقى قائمة بمصادر البيانات اللي تم إرسالها فعلاً للموديل
        used_data_sources = []
        context_data = []

        # ***** التعديل هنا: استخدام selected_sources لتحديد أي بيانات يتم جلبها *****
        # تأكد إنك بتشيك على selected_sources عشان لو فاضية أو مش موجودة
        sources_to_include = set(query.selected_sources) if query.selected_sources else set()

        if "users" in sources_to_include:
            users = db.query(DBUser).all()
            context_data.append({
                "source": "users",
                "data": [{
                    "id": u.user_id,
                    "email": u.email,
                    "role": u.role.value if u.role else None,
                    "first_name": u.first_name,
                    "last_name": u.last_name,
                    "created_at": u.created_at.isoformat() if u.created_at else None,
                    "governorate": u.governorate.value if u.governorate else None,
                    "lawyer_balance": u.lawyer_balance if u.role == RoleEnum.lawyer else None,
                    "specialization": u.specialization if u.role == RoleEnum.lawyer else None,
                    "license_number": u.license_number if u.role == RoleEnum.lawyer else None
                } for u in users]
            })
            used_data_sources.append("users")

        if "subscriptions" in sources_to_include:
            subscriptions = db.query(Subscription).all()
            context_data.append({
                "source": "subscriptions",
                "data": [{
                    "id": s.sub_id,
                    "user_id": s.user_id,
                    "plan": s.plan_name.value if s.plan_name else None,
                    "price": s.price,
                    "start_date": s.start_date.isoformat() if s.start_date else None,
                    "end_date": s.end_date.isoformat() if s.end_date else None,
                    "status": s.status,
                    "paymob_subscription_id": s.paymob_subscription_id
                } for s in subscriptions]
            })
            used_data_sources.append("subscriptions")

        if "payments" in sources_to_include:
            payments = db.query(Payment).all()
            context_data.append({
                "source": "payments",
                "data": [{
                    "id": p.payment_id,
                    "amount": p.amount,
                    "currency": p.currency,
                    "status": p.payment_status.value if p.payment_status else None,
                    "method": p.payment_method,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "subscription_id": p.sub_id
                } for p in payments]
            })
            used_data_sources.append("payments")

        if "chats" in sources_to_include:
            chats = db.query(Chat).all()
            context_data.append({
                "source": "chats",
                "data": [{
                    "id": c.chat_id,
                    "type": c.chat_type.value if c.chat_type else None,
                    "status": c.status.value if c.status else None,
                    "started_at": c.started_at.isoformat() if c.started_at else None,
                    "ended_at": c.ended_at.isoformat() if c.ended_at else None,
                    "auto_close_at": c.auto_close_at.isoformat() if c.auto_close_at else None
                } for c in chats]
            })
            used_data_sources.append("chats")

        if "message_stats" in sources_to_include:
            message_stats = db.query(
                Queries.chat_id,
                func.count(Queries.query_id).label('message_count'),
                func.min(Queries.created_at).label('first_message'),
                func.max(Queries.created_at).label('last_message')
            ).group_by(Queries.chat_id).all()
            context_data.append({
                "source": "message_stats",
                "data": [{
                    "chat_id": stat.chat_id,
                    "message_count": stat.message_count,
                    "first_message": stat.first_message.isoformat() if stat.first_message else None,
                    "last_message": stat.last_message.isoformat() if stat.last_message else None
                } for stat in message_stats]
            })
            used_data_sources.append("message_stats")

        if "withdrawal_requests" in sources_to_include:
            withdrawal_requests = db.query(WithdrawalRequest).all()
            context_data.append({
                "source": "withdrawal_requests",
                "data": [{
                    "id": w.request_id,
                    "lawyer_id": w.lawyer_id,
                    "amount": w.amount,
                    "phone_number": w.phone_number,
                    "status": w.status.value if w.status else None,
                    "created_at": w.created_at.isoformat() if w.created_at else None,
                    "updated_at": w.updated_at.isoformat() if w.updated_at else None
                } for w in withdrawal_requests]
            })
            used_data_sources.append("withdrawal_requests")

        if "message_quotas" in sources_to_include:
            message_quotas = db.query(MessageQuota).all()
            context_data.append({
                "source": "message_quotas",
                "data": [{
                    "user_id": q.user_id,
                    "remaining_messages": q.remaining_messages,
                    "total_messages": q.total_messages,
                    "expiry_date": q.expiry_date.isoformat() if q.expiry_date else None
                } for q in message_quotas]
            })
            used_data_sources.append("message_quotas")

        admin_template = """
        You are an administrative assistant specialized in analyzing system data.
        Use the following information to answer the question.
        Analyze the available data and provide a comprehensive and detailed response.
        IMPORTANT: Please respond in Arabic language.

        Available data:
        {context}

        Question: {question}

        Detailed response:
        """

        prompt = admin_template.format(
            context=json.dumps(context_data, ensure_ascii=False),
            question=query.question
        )

        # ****** هنا الجزء اللي بيعمل call للموديل الخارجي، وده ممكن يكون سبب الـ timeout ******
        # تأكد إن الـ client ده بيتعامل مع timeouts صح
        # لو المشكلة مستمرة، هتحتاج تزود الـ timeout هنا في استدعاء الـ client.chat.completions.create
        # أو في إعدادات الـ client نفسه (لو الـ library بتدعم)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            # ممكن تضيف timeout هنا لو الـ library بتدعم، مثلاً:
            # timeout=30.0 # 30 ثانية، او حسب احتياجك
        )
        response = completion.choices[0].message.content.strip() # ملاحظة: الوصول للمحتوى ممكن يختلف شوية حسب إصدار الـ library


        processing_time = (datetime.utcnow() - start_time).total_seconds()

        if current_user:
            new_query = Queries(
                query_text=query.question,
                user_id=current_user.user_id,
                chat_id=query.chat_id
            )
            db.add(new_query)
            db.flush()

            new_response = Responses(
                responses_text=response,
                generated_by="admin_model",
                query_id=new_query.query_id,
                chat_id=query.chat_id
            )
            db.add(new_response)
            db.flush()
            db.commit()

        return AdminModelResponse(
            response=response,
            processing_time=processing_time,
            tokens_used=len(response.split()), # تقدير لعدد الـ tokens
            confidence_score=0.9,
            data_sources=used_data_sources # بنرجع مصادر البيانات اللي استخدمناها فعلاً
        )

    except Exception as e:
        logging.error(f"Error in admin model chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    

@app.get("/admin/model/chat/history", response_model=List[AdminModelResponse])
async def get_admin_chat_history(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 20,
    offset: int = 0
):
    """Get admin model chat history"""
    if current_user.role != RoleEnum.admin:
        raise HTTPException(
            status_code=403,
            detail="Only admin can access this endpoint"
        )

    chats = db.query(Chat).filter(
        Chat.chat_type == ChatTypeEnum.model_chat
    ).order_by(
        Chat.started_at.desc()
    ).offset(offset).limit(limit).all()

    history = []
    for chat in chats:
        queries = db.query(Queries).filter(
            Queries.chat_id == chat.chat_id
        ).order_by(Queries.created_at).all()

        for query in queries:
            response = db.query(Responses).filter(
                Responses.query_id == query.query_id
            ).first()

            if response:
                history.append(AdminModelResponse(
                    response=response.responses_text,
                    processing_time=0.0,
                    tokens_used=len(response.responses_text.split()),
                    confidence_score=0.9,
                    data_sources=[]
                ))

    return history


# ==================================================
# Support Chat APIs
# ==================================================

@app.get("/admin/support/chats", response_model=List[SupportChat])
def get_support_chats(
    user_type: Optional[SupportChatType] = None,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get support chats (admin only)"""
    try:
        check_admin_access(current_user)

        close_expired_chats(db)

        query = db.query(SupportChat).join(DBUser, SupportChat.user_id == DBUser.user_id)

        if user_type:
            query = query.filter(DBUser.role == user_type)

        chats = query.order_by(SupportChat.updated_at.desc()).all()

        result = []
        for chat in chats:
            user = db.query(DBUser).filter(DBUser.user_id == chat.user_id).first()
            if not user:
                continue

            last_message = db.query(SupportMessage).filter(
                SupportMessage.chat_id == chat.chat_id
            ).order_by(SupportMessage.created_at.desc()).first()

            unread_count = db.query(SupportMessage).filter(
                SupportMessage.chat_id == chat.chat_id,
                SupportMessage.is_read == False,
                SupportMessage.sender_id != current_user.user_id
            ).count()

            result.append(SupportChat(
                chat_id=chat.chat_id,
                user_id=user.user_id,
                user_name=f"{user.first_name} {user.last_name}",
                user_email=user.email,
                user_type=user.role,
                last_message=last_message.content if last_message else None,
                last_message_time=last_message.created_at if last_message else None,
                unread_count=unread_count,
                status=chat.status,
                created_at=chat.created_at
            ))

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


class SupportChatResponse(BaseModel):
    chat_id: int

@app.post("/support/chat", response_model=SupportChatResponse)
def create_support_chat(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # 1. التحقق من الحد الأقصى للمحادثات النشطة (3 محادثات)
        active_support_chats_count = db.query(Chat).filter(
            Chat.participants.contains([current_user.user_id]),
            Chat.status == ChatStatusEnum.active,
            Chat.chat_type == ChatTypeEnum.support # فلترة صريحة حسب نوع الشات
        ).count()

        if active_support_chats_count >= 3:
            raise HTTPException(
                status_code=400,
                detail="لقد وصلت إلى الحد الأقصى من محادثات الدعم النشطة (3 محادثات). برجاء إنهاء محادثة قائمة قبل بدء محادثة جديدة."
            )

        # 2. التحقق مما إذا كان المستخدم لديه أي اشتراك سابق (نشط أو غير نشط)
        user_has_subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.user_id
        ).first()

        if not user_has_subscription:
            raise HTTPException(
                status_code=403, # 403 Forbidden مناسب لعدم وجود اشتراك
                detail="يجب أن يكون لديك اشتراك سابق (نشط أو غير نشط) لبدء محادثة دعم."
            )

        # 3. إنشاء محادثة دعم جديدة دائمًا (لا نتحقق من وجود محادثة سابقة لإرجاعها)
        new_chat = Chat(
            participants=[current_user.user_id, 100], # 100 هو ID الادمن 
            status=ChatStatusEnum.active,
            chat_type=ChatTypeEnum.support, # نوع الشات هو دعم فني
            started_at=datetime.utcnow(), # يتم تعيين تاريخ البدء
            auto_close_at=datetime.utcnow() + timedelta(hours=48) # مثال: تغلق بعد 48 ساعة
        )
        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)

        # 4. إرجاع الـ chat_id الخاص بالمحادثة الجديدة فقط
        return SupportChatResponse(chat_id=new_chat.chat_id)

    except HTTPException:
        # في حالة وجود HTTPException، قم برفعها مباشرة
        raise
    except Exception as e:
        # لأي أخطاء أخرى غير متوقعة، قم بعمل rollback ورفع HTTP 500
        db.rollback()
        logging.error(f"Internal server error while creating support chat for user {current_user.user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"حدث خطأ داخلي أثناء إنشاء محادثة الدعم: {str(e)}"
        )

def close_expired_chats(db: Session):

    """Close only expired model chats"""

    expired_chats = db.query(Chat).filter(

        Chat.status == ChatStatusEnum.active,

        Chat.chat_type == ChatTypeEnum.model_chat,

        Chat.auto_close_at <= datetime.utcnow()

    ).all()



    for chat in expired_chats:

        chat.status = ChatStatusEnum.inactive

        chat.ended_at = datetime.utcnow()

        db.add(chat)



    db.commit()



@app.post("/support/chat/{chat_id}/message", response_model=SuccessResponse)
def send_support_message(
    chat_id: int,
    content: str,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message in a support chat"""
    chat = db.query(Chat).filter(
        Chat.chat_id == chat_id,
        Chat.status == ChatStatusEnum.active,
        Chat.auto_close_at > datetime.utcnow()
    ).first()

    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Active chat not found or has expired"
        )

    if current_user.role != RoleEnum.admin and current_user.user_id not in chat.participants:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to send messages in this chat"
        )


    # --- Conditional Logic for Sending Query vs. Response ---
    if current_user.role == RoleEnum.admin:
        # If the current user is an admin, they are sending a Response
        last_query = db.query(Queries).filter(
            Queries.chat_id == chat_id
        ).order_by(Queries.created_at.desc()).first()

        if not last_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot send a response: No previous queries found in this chat."
            )

        new_message = Responses(
            responses_text=content, # Use content for response text
            generated_by=f"lawyer_{current_user.user_id}" if current_user.role == RoleEnum.lawyer else f"admin_{current_user.user_id}", # Or just admin if only admin can send responses
            query_id=last_query.query_id,
            chat_id=chat_id,
            created_at=datetime.utcnow()
        )
        message_type = "response"

    else:
        # If the current user is NOT an admin (e.g., a regular user), they are sending a Query
        new_message = Queries(
            query_text=content,
            user_id=current_user.user_id,
            chat_id=chat_id,
            created_at=datetime.utcnow()
        )
        message_type = "query"

    db.add(new_message)
    db.commit()
    db.refresh(new_message) # Refresh to get auto-generated IDs and timestamps

    return SuccessResponse(message=f"{message_type.capitalize()} sent successfully.")


@app.put("/support/chat/{chat_id}/end", response_model=SupportChat)
def end_support_chat(
    chat_id: int,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """End a support chat"""
    chat = db.query(SupportChat).filter(SupportChat.chat_id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if current_user.role != RoleEnum.admin and chat.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to end this chat")

    chat.status = ChatStatusEnum.inactive
    db.add(chat)
    db.commit()
    db.refresh(chat)

    return chat

def close_expired_chats_job():
    """Background job to close expired chats"""
    db = SessionLocal()
    try:
        close_expired_chats(db)
    finally:
        db.close()


nest_asyncio.apply()




@app.post("/model/chat", response_model=ModelResponse)
async def model_chat(
    query: ModelQuery,
    user_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    """Chat with the model, supporting user files and legal references."""
    try:
        start_time = datetime.utcnow()

        # Validate chat existence and status
        chat = db.query(Chat).filter(
            Chat.chat_id == query.chat_id,
            Chat.chat_type == ChatTypeEnum.model_chat,
            Chat.status == ChatStatusEnum.active
        ).first()

        if not chat:
            raise HTTPException(
                status_code=404,
                detail="Chat not found or has ended"
            )

        # Retrieve legal reference files
        legal_data = load_legal_files()
        relevant_file = identify_relevant_file(query.question, legal_data)

        # Process uploaded user file
        user_file_content = ""
        if user_file:
            try:
                file_content = await user_file.read()
                user_file_content = file_content.decode("utf-8")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing user file: {str(e)}"
                )
# Extract text based on file type
        file_extension = UploadFile.filename.split('.')[-1].lower()
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file_content)
        elif file_extension == 'docx':
            text = extract_text_from_docx(file_content)
        elif file_extension == 'txt':
            text = extract_text_from_txt(file_content)
        elif file_extension in ['xlsx', 'xls']:
            text = extract_text_from_excel(file_content)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Supported formats: PDF, DOCX, TXT, Excel"
            )
        # Build the context for the model
        question_language = detect_language(query.question)

        context = f"""You are a professional legal assistant with deep knowledge of Egyptian law.
You are confident in your answers and provide comprehensive, easy-to-understand explanations.

Important: Please respond in the same language as the question. The question is in {question_language}.

"""

        if relevant_file:
            context += f"Relevant legal context from {relevant_file}:\n"
            context += json.dumps(legal_data.get(relevant_file, {}), ensure_ascii=False, indent=2)

        if user_file_content:
            context += f"\nUser-provided file content (file name: {user_file.filename}):\n"
            context += user_file_content

        context += f"""
Current question: {query.question}

Please provide a detailed, clear, and accurate response to this specific question based on the provided legal context, user file, and your expertise. If the user file is irrelevant to the question, focus solely on the legal context and question. If no legal context matches the question, respond using only your general expertise.
"""

        # Call the model to get a response
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": context}]
        )
        model_response = completion.choices[0].message["content"].strip()

        # Log the question
        new_query = Queries(
            query_text=query.question,
            user_id=current_user.user_id,
            chat_id=chat.chat_id,
            status="sent"
        )
        db.add(new_query)
        db.flush()

        # Log the response
        new_response = Responses(
            responses_text=model_response,
            generated_by="model",
            query_id=new_query.query_id,
            chat_id=chat.chat_id
        )
        db.add(new_response)

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        db.commit()

        return ModelResponse(
            response=model_response,
            processing_time=processing_time,
            tokens_used=len(model_response.split()),
            confidence_score=0.8,
            chat_id=chat.chat_id,
            query_id=new_query.query_id
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/model/chat/{chat_id}/history", response_model=ModelChatHistory)
async def get_model_chat_history(
    chat_id: int,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get model chat history"""
    chat = db.query(Chat).filter(
        Chat.chat_id == chat_id,
        Chat.chat_type == ChatTypeEnum.model_chat
    ).first()

    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Chat not found"
        )

    # Check permissions
    is_owner = any(
        q.user_id == current_user.user_id
        for q in chat.queries
    )

    if not is_owner and current_user.role != RoleEnum.admin:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this chat"
        )

    # Get all messages
    messages = []
    queries = db.query(Queries).filter(
        Queries.chat_id == chat_id
    ).order_by(Queries.created_at).all()

    for query in queries:
        messages.append({
            "type": "user",
            "content": query.query_text,
            "timestamp": query.created_at
        })

        responses = db.query(Responses).filter(
            Responses.query_id == query.query_id
        ).order_by(Responses.created_at).all()

        for response in responses:
            messages.append({
                "type": "model",
                "content": response.responses_text,
                "timestamp": response.created_at
            })

    return ModelChatHistory(
        chat_id=chat.chat_id,
        messages=messages,
        status=chat.status,
        auto_close_at=chat.auto_close_at
    )

@app.post("/model/chat/{chat_id}/end", response_model=ChatResponse)
async def end_model_chat(
    chat_id: int,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """End model chat"""
    chat = db.query(Chat).filter(
        Chat.chat_id == chat_id,
        Chat.chat_type == ChatTypeEnum.model_chat,
        Chat.status == ChatStatusEnum.active
    ).first()

    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Chat not found or already ended"
        )

    # Check permissions
    is_owner = any(
        q.user_id == current_user.user_id
        for q in chat.queries
    )

    if not is_owner and current_user.role != RoleEnum.admin:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to end this chat"
        )

    chat.status = ChatStatusEnum.inactive
    chat.ended_at = datetime.utcnow()
    db.add(chat)
    db.commit()

    return chat

@app.post("/model/orders", response_model=ModelOrderResponse)
async def create_model_order(
    order: ModelOrderCreate,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create new model order"""
    if current_user.role != RoleEnum.company:
        raise HTTPException(
            status_code=403,
            detail="Only companies can create model orders"
        )

    new_order = ModelOrderDB(
        company_id=current_user.user_id,
        order_type=order.order_type,
        description=order.description,
        database_schema=order.database_schema,
        training_data=order.training_data
    )

    db.add(new_order)
    db.commit()
    db.refresh(new_order)

    return new_order

@app.get("/model/orders", response_model=List[ModelOrderResponse])
async def get_model_orders(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    status: Optional[ModelOrderStatus] = None
):
    """Get list of model orders"""
    query = db.query(ModelOrderDB)

    if current_user.role == RoleEnum.company:
        query = query.filter(ModelOrderDB.company_id == current_user.user_id)

    if status:
        query = query.filter(ModelOrderDB.status == status)

    orders = query.order_by(ModelOrderDB.created_at.desc()).all()
    return orders

@app.get("/model/orders/{order_id}", response_model=ModelOrderResponse)
async def get_model_order(
    order_id: int,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get model order details"""
    order = db.query(ModelOrderDB).filter(ModelOrderDB.order_id == order_id).first()

    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    if current_user.role == RoleEnum.company and order.company_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this order")

    return order

@app.put("/model/orders/{order_id}/status", response_model=ModelOrderResponse)
async def update_order_status(
    order_id: int,
    status: ModelOrderStatus,
    admin_notes: Optional[str] = None,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update model order status (admin only)"""
    if current_user.role != RoleEnum.admin:
        raise HTTPException(status_code=403, detail="Only admin can update order status")

    order = db.query(ModelOrderDB).filter(ModelOrderDB.order_id == order_id).first()

    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    order.status = status
    order.admin_notes = admin_notes

    if status == ModelOrderStatus.completed:
        order.completed_at = datetime.utcnow()

    db.add(order)
    db.commit()
    db.refresh(order)

    return order


# ==================================================
# model Chat APIs
# ==================================================

@app.post("/chatttt", response_model=MChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: Optional[DBUser] = Depends(get_current_user)
):
    try:
        chat_id = request.chat_id or f"chat_{datetime.now().timestamp()}"

        chat_history_summary = ""
        if current_user:
            summaries = db.query(ChatSummary).join(Queries).filter(
                Queries.chat_id == chat_id,
                Queries.user_id == current_user.user_id
            ).order_by(ChatSummary.created_at).all()

            chat_history_summary = "\n".join([
                f"Q: {s.question_summary}\nA: {s.answer_summary}" for s in summaries
            ])
        else:
            chat_history = user_chats.get(chat_id, [])
            chat_history_summary = format_chat_history(chat_history)

        question_language = detect_language(request.message)

        legal_data = load_legal_files()
        relevant_file = identify_relevant_file(request.message, legal_data)

        context = f"""You are a professional legal assistant with deep knowledge of Egyptian law.
You are confident in your answers and provide comprehensive, easy-to-understand explanations.

Important: Please respond in the same language as the question. The question is in {question_language}.

Relevant legal context from {relevant_file}:
{json.dumps(legal_data.get(relevant_file, {}), ensure_ascii=False, indent=2)}

Previous conversation (for context only - do not repeat or answer previous questions):
{chat_history_summary}

Give special attention to the **last message in the previous conversation**, as the current question is likely related to it. Use it primarily for understanding the immediate context, but do not repeat its content or answer it directly.

Only answer the **Current question** below using the legal context and conversation history for background understanding, but do not reference or re-answer previous questions.

Current question: {request.message}

Please provide a detailed, clear, and accurate response to this specific question based on the provided legal context and your expertise.
"""

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": context}]
        )
        response = completion.choices[0].message["content"].strip()

        summarize_prompt = lambda text: f"""
You must strictly summarize the input text without adding, interpreting, or expanding on the content in any way. Your only task is to restate the same meaning as briefly as possible, while preserving the core message exactly.

If the input is a long paragraph, rewrite it in a shorter form that conveys the same meaning clearly and accurately.

If the input is already short, rewrite it exactly as it is, or rephrase it slightly without changing the meaning.

If the input is a question, do not answer it — simply rewrite the question in a more concise and clear way.

If the input is an answer, summarize it by extracting the main points only, without changing the intended meaning.

Absolutely do not add any new ideas, interpretations, suggestions, or conclusions under any circumstances.

Be precise, faithful to the original content with the same language, and focused only on rewriting or summarizing—never generate new information or responses
the input text:\n{text}"""

        question_summary = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": summarize_prompt(request.message)}]
        ).choices[0].message["content"].strip()

        answer_summary = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": summarize_prompt(response)}]
        ).choices[0].message["content"].strip()

        if current_user:
            new_query = Queries(
                query_text=request.message,
                user_id=current_user.user_id,
                chat_id=chat_id
            )
            db.add(new_query)
            db.flush()

            new_response = Responses(
                responses_text=response,
                generated_by="model",
                query_id=new_query.query_id,
                chat_id=chat_id
            )
            db.add(new_response)
            db.flush()

            summary = ChatSummary(
                query_id=new_query.query_id,
                response_id=new_response.responses_id,
                question_summary=question_summary,
                answer_summary=answer_summary
            )
            db.add(summary)
            db.commit()
        else:
            chat_history = user_chats.get(chat_id, [])
            chat_history.append(MChatMessage(role="user", content=request.message))
            chat_history.append(MChatMessage(role="assistant", content=response))
            user_chats[chat_id] = chat_history

        return MChatResponse(response=response, chat_id=chat_id)

    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )






# Paymob Integration
PAYMOB_API_KEY = "ZXlKaGJHY2lPaUpJVXpVeE1pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmpiR0Z6Y3lJNklrMWxjbU5vWVc1MElpd2ljSEp2Wm1sc1pWOXdheUk2TVRBME1UTTBPU3dpYm1GdFpTSTZJbWx1YVhScFlXd2lmUS5mYWVQck9TbnJCa2lSTjZZWGNOaHhaLXRSakRTSHE2UFozU1lDRV9UZWJCUm54aEN2THpsSV9QWDcyV3NNUDRtcUN6RlZkQ1RWbjJXUGNTeVdwdEQxQQ=="
PAYMOB_INTEGRATION_ID = 5076904
PAYMOB_IFRAME_ID = "918243"
PAYMOB_HMAC_SECRET = "2FFDE4FFB1D3759E0AC3ADA3EDA26FF3"



async def get_paymob_auth_token():
    url = "https://accept.paymob.com/api/auth/tokens"
    payload = {"api_key": PAYMOB_API_KEY}
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("token")

async def create_paymob_order(auth_token: str, amount_cents: int, merchant_order_id: str):
    url = "https://accept.paymob.com/api/ecommerce/orders"
    payload = {
        "auth_token": auth_token,
        "delivery_needed": "false",
        "amount_cents": amount_cents,
        "currency": "EGP",
        "merchant_order_id": merchant_order_id,
        "items": []
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("id")

async def generate_payment_key(auth_token: str, amount_cents: int, order_id: str, user_data: dict):
    url = "https://accept.paymob.com/api/acceptance/payment_keys"
    payload = {
        "auth_token": auth_token,
        "amount_cents": amount_cents,
        "expiration": 3600,
        "order_id": order_id,
        "billing_data": {
            "apartment": "NA",
            "email": user_data.get("email", "user@example.com"),
            "floor": "NA",
            "first_name": user_data.get("first_name", "User"),
            "street": "NA",
            "building": "NA",
            "phone_number": user_data.get("phone", "+201234567890"),
            "shipping_method": "NA",
            "postal_code": "NA",
            "city": "NA",
            "country": "NA",
            "last_name": user_data.get("last_name", "NA"),
            "state": "NA"
        },
        "currency": "EGP",
        "integration_id": PAYMOB_INTEGRATION_ID
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("token")

@app.post("/payments/initiate", response_model=PaymentResponse)
async def initiate_payment(
    payment_request: PaymentRequest,
    current_user: DBUser = Depends(get_current_user),  # Require authentication
    db: Session = Depends(get_db)
):
    try:
        plan_details = get_plan_details(payment_request.plan_name)
        amount_cents = plan_details["price"] * 100
        merchant_order_id = f"user_{payment_request.user_id}_{int(datetime.now().timestamp())}"
        auth_token = await get_paymob_auth_token()
        order_id = await create_paymob_order(auth_token, amount_cents, merchant_order_id)
        payment_key = await generate_payment_key(
            auth_token=auth_token,
            amount_cents=amount_cents,
            order_id=order_id,
            user_data={
                "email": current_user.email,
                "first_name": current_user.first_name,
                "last_name": current_user.last_name,
                "phone": current_user.phone_number or "+201234567890"
            }
        )
        payment_url = f"https://accept.paymob.com/api/acceptance/iframes/{PAYMOB_IFRAME_ID}?payment_token={payment_key}"
        return PaymentResponse(payment_url=payment_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def get_field(obj, field_path):
    parts = field_path.split(".")
    value = obj
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part, "")
        else:
            return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(int(value)) if value == int(value) else str(value)
    return str(value) if value is not None else ""


# Assuming these are defined elsewhere in your code

def extend_subscription_period(current_end_date: datetime, plan_name: str) -> datetime:
    """
    Extend the subscription end date based on the plan type.

    Args:
        current_end_date (datetime): Current subscription end date
        plan_name (str): Plan type (daily, monthly, annually)

    Returns:
        datetime: New end date
    """
    plan_durations = {
        "daily": 1,      
        "monthly": 30,  
        "annually": 365    
    }

    duration_days = plan_durations.get(plan_name, 30)  
    return current_end_date + timedelta(days=duration_days)

@app.post("/payments/webhook")
async def paymob_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        # 1. Get raw body and query parameters
        body_bytes = await request.body()
        received_hmac = request.query_params.get("hmac")
        if not received_hmac:
            print("HMAC signature missing")
            return JSONResponse(status_code=400, content={"error": "HMAC signature missing"})

        # 2. Parse JSON data
        try:
            data = await request.json()
        except Exception as e:
            print(f"Invalid JSON: {str(e)}")
            return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

        # 3. Verify this is a transaction notification
        if data.get("type") != "TRANSACTION":
            print("Ignoring non-transaction notification")
            return JSONResponse(status_code=200, content={"status": "ignored"})

        transaction = data.get("obj", {})

        # 4. Generate HMAC string
        hmac_fields = [
            "amount_cents", "created_at", "currency", "error_occured", "has_parent_transaction",
            "id", "integration_id", "is_3d_secure", "is_auth", "is_capture", "is_refunded",
            "is_standalone_payment", "is_voided", "order.id", "owner", "pending",
            "source_data.pan", "source_data.sub_type", "source_data.type", "success"
        ]
        hmac_values = []
        for field in hmac_fields:
            value = get_field(transaction, field)
            hmac_values.append(value)
            print(f"Field {field}: {value}")
        hmac_string = "".join(hmac_values)
        print(f"HMAC String: {hmac_string}")

        # 5. Calculate HMAC
        calculated_hmac = hmac.new(
            PAYMOB_HMAC_SECRET.encode("utf-8"),
            hmac_string.encode("utf-8"),
            hashlib.sha512
        ).hexdigest()
        print(f"Received HMAC: {received_hmac}")
        print(f"Calculated HMAC: {calculated_hmac}")

        # 6. Verify HMAC
        if not hmac.compare_digest(received_hmac.encode("utf-8"), calculated_hmac.encode("utf-8")):
            print("HMAC verification failed")
            return JSONResponse(status_code=403, content={"error": "Invalid HMAC signature"})

        # 7. Only process successful transactions
        if not transaction.get("success", False):
            print("Transaction not successful, ignoring")
            return JSONResponse(status_code=200, content={"status": "ignored"})

        # 8. Handle recurring subscription
        if "subscription_id" in transaction:
            subscription = db.query(Subscription).filter(
                Subscription.paymob_subscription_id == transaction["subscription_id"]
            ).first()
            if subscription:
                subscription.end_date = extend_subscription_period(
                    subscription.end_date,
                    subscription.plan_name
                )
                db.commit()
                print(f"Extended subscription {subscription.sub_id}")

        # 9. Extract user ID from merchant_order_id
        merchant_order_id = transaction.get("order", {}).get("merchant_order_id", "")
        try:
            _, user_id_str, _ = merchant_order_id.split("_")
            user_id = int(user_id_str)
        except Exception as e:
            print(f"Invalid merchant_order_id format: {merchant_order_id}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid merchant_order_id format"}
            )

        amount_cents = transaction.get("amount_cents", 0)
        amount = amount_cents / 100  # Convert cents to EGP (pounds)
        if amount_cents == 20000:  # 200 EGP
            plan_name = PlanNameEnum.daily
        elif amount_cents == 200000:  # 2000 EGP
            plan_name = PlanNameEnum.monthly
        elif amount_cents == 2000000:  # 20000 EGP
            plan_name = PlanNameEnum.annually
        else:
            print(f"Unknown payment amount: {amount} EGP ({amount_cents} cents)")
            return JSONResponse(
                status_code=400,
                content={"error": f"Unknown payment amount: {amount} EGP"}
            )

        # 11. Check for existing subscription
        existing_sub = db.query(Subscription).filter(
            Subscription.user_id == user_id,
            Subscription.plan_name == plan_name,
            Subscription.status == "active",
            Subscription.end_date >= datetime.utcnow()
        ).first()
        if existing_sub:
            print(f"Subscription {existing_sub.sub_id} already exists")
            return JSONResponse(status_code=200, content={"status": "already_exists"})

        # 12. Create new subscription
        plan_details = get_plan_details(plan_name)
        start_date = datetime.utcnow()
        end_date = start_date + timedelta(days=plan_details["duration_days"])
        subscription = Subscription(
            plan_name=plan_name,
            price=plan_details["price"],
            start_date=start_date,
            end_date=end_date,
            status="active",
            user_id=user_id
        )
        db.add(subscription)
        db.flush()

        # 13. Create payment record
        payment_method = transaction.get("source_data", {}).get("type", "card")
        payment = Payment(
            amount=amount,
            currency="EGP",
            payment_status=PaymentStatusEnum.successful,
            payment_method=payment_method,
            sub_id=subscription.sub_id,
            created_at=datetime.utcnow()
        )
        db.add(payment)
        db.commit()

        print(f"Successfully created subscription {subscription.sub_id} for user {user_id}")
        return JSONResponse(status_code=200, content={"status": "success"})

    except Exception as e:
        db.rollback()
        print(f"Webhook processing failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})



async def handle_failed_payment(subscription_id: str, db: Session):
    subscription = db.query(Subscription).filter(
        Subscription.paymob_subscription_id == subscription_id
    ).first()
    if subscription.retry_count < 3:
        await retry_payment(subscription, db)
    else:
        subscription.status = "suspended"
        db.commit()
        notify_admin(subscription)

@app.post("/subscriptions/cancel")
async def cancel_subscription(
    subscription_id: int,
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)  # Use Depends to inject the resolved user
):
    try:
        # Fetch the subscription
        subscription = db.query(Subscription).filter(Subscription.sub_id == subscription_id).first()
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")

        # Check if the user has permission
        if subscription.user_id != current_user.user_id and current_user.role != RoleEnum.admin:
            raise HTTPException(status_code=403, detail="Unauthorized")

        # Update subscription status
        old_status = subscription.status
        subscription.status = "inactive"
        db.commit()
        print(f"Subscription {subscription_id} updated from {old_status} to inactive for user {current_user.user_id}")

        return {"message": "Subscription cancelled successfully", "subscription_id": subscription_id}
    except HTTPException as e:
        raise e
    except Exception as e:
        db.rollback()
        print(f"Failed to cancel subscription {subscription_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel subscription: {str(e)}")


async def retry_payment(subscription: Subscription, db: Session):
    try:
        auth_token = await get_paymob_auth_token()
        url = "https://accept.paymob.com/api/subscription/retry"
        payload = {
            "auth_token": auth_token,
            "subscription_id": subscription.paymob_subscription_id
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                subscription.retry_count = 0
            else:
                subscription.retry_count += 1
            db.commit()
    except Exception as e:
        print(f"Retry payment failed: {str(e)}")
        db.rollback()

def notify_admin(subscription):
    print(f"Admin alert: Subscription {subscription.sub_id} suspended after 3 retries")




# @app.get("/chat/{chat_id}")
# async def get_chat_history(chat_id: str):
#     """Get specific chat history"""
#     try:
#         if chat_id not in user_chats:
#             raise HTTPException(status_code=404, detail="Chat not found")
#         return user_chats[chat_id]
#     except Exception as e:
#         print(f"Error in get_chat_history: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"An error occurred while retrieving chat history: {str(e)}"
#         )



# ==================================================
# Lawyer APIs
# ==================================================

class LawyerResponse(BaseModel):
    user_id: int
    first_name: str
    last_name: str
    email: str
    phone_number: str
    specialization: str
    governorate: GovernorateEnum
    lawyer_balance: float

    class Config:
        from_attributes = True

@app.get("/admin/lawyers", response_model=List[LawyerResponse])
def get_all_lawyers(
    db: Session = Depends(get_db)
):
    """Get all lawyers"""
    lawyers = db.query(DBUser).filter(
        DBUser.role == RoleEnum.lawyer
    ).all()
    return lawyers


@app.get("/lawyers/by-governorate/{governorate}", response_model=List[LawyerResponse])
def get_lawyers_by_governorate(
    governorate: GovernorateEnum,
    db: Session = Depends(get_db)
):
    """Get lawyers in specific governorate"""
    # Get lawyers in the same governorate
    local_lawyers = db.query(DBUser).filter(
        DBUser.role == RoleEnum.lawyer,
        DBUser.governorate == governorate
    ).all()

    if not local_lawyers:
        # If no lawyers in the governorate, get lawyers from other governorates
        other_lawyers = db.query(DBUser).filter(
            DBUser.role == RoleEnum.lawyer,
            DBUser.governorate != governorate
        ).all()

        if not other_lawyers:
            raise HTTPException(
                status_code=404,
                detail="No lawyers available at the moment"
            )

        return other_lawyers

    return local_lawyers

@app.get("/lawyers/balance", response_model=float)
def get_lawyer_balance(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get lawyer's current balance"""
    if current_user.role != RoleEnum.lawyer:
        raise HTTPException(
            status_code=403,
            detail="Only lawyers can access their balance"
        )

    return current_user.lawyer_balance

@app.get("/governorates", response_model=List[str])
def get_governorates():
    """Get list of all Egyptian governorates"""
    return [gov.value for gov in GovernorateEnum]

class DocumentSummaryRequest(BaseModel):
    chat_id: int
    temperature: float = 0.7
    max_tokens: int = 1000

class DocumentSummaryResponse(BaseModel):
    summary: str
    processing_time: float
    tokens_used: int
    confidence_score: float

class CaseAnalysisRequest(BaseModel):
    chat_id: int
    temperature: float = 0.7
    max_tokens: int = 1000

class CaseAnalysisResponse(BaseModel):
    analysis: str
    processing_time: float
    tokens_used: int
    confidence_score: float

def extract_text_from_pdf(file_content: bytes) -> str:
    pdf_file = io.BytesIO(file_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file_content: bytes) -> str:
    docx_file = io.BytesIO(file_content)
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(file_content: bytes) -> str:
    return file_content.decode('utf-8')

def extract_text_from_excel(file_content: bytes) -> str:
    excel_file = io.BytesIO(file_content)
    df = pd.read_excel(excel_file)
    return df.to_string()

@app.post("/document/summarize", response_model=DocumentSummaryResponse)
async def summarize_document(
    file: UploadFile = File(...),
    request: DocumentSummaryRequest = None,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Summarize document content using AI model"""
    try:
        start_time = datetime.utcnow()

        # Read file content
        file_content = await file.read()

        # Extract text based on file type
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file_content)
        elif file_extension == 'docx':
            text = extract_text_from_docx(file_content)
        elif file_extension == 'txt':
            text = extract_text_from_txt(file_content)
        elif file_extension in ['xlsx', 'xls']:
            text = extract_text_from_excel(file_content)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Supported formats: PDF, DOCX, TXT, Excel"
            )

        # Prepare prompt for summarization
        prompt = f"""
        You are a professional document summarizer. Your task is to provide a detailed and accurate summary of the following content.
        Focus on maintaining the key points and important information while organizing the content in a clear and logical structure.

        Content to summarize:
        {text}

        Please provide a comprehensive summary that:
        1. Captures all important points
        2. Maintains the original context and meaning
        3. Is well-organized and easy to understand
        4. Preserves any critical data or statistics
        """

        # Get model response
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        summary = completion.choices[0].message["content"].strip()

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return DocumentSummaryResponse(
            summary=summary,
            processing_time=processing_time,
            tokens_used=len(summary.split()),
            confidence_score=0.9
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@app.post("/case/analyze", response_model=CaseAnalysisResponse)
async def analyze_case(
    file: UploadFile = File(...),
    request: CaseAnalysisRequest = None,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze case details from the uploaded file and provide a comprehensive legal analysis.
    The endpoint attempts to:
    1. Understand the nature of the case.
    2. Identify potential legal defenses or solutions.
    3. Generate results exclusively in Arabic, maintaining clarity and accuracy.

    Supported file formats: PDF, DOCX, TXT, Excel.
    """
    try:
        start_time = datetime.utcnow()

        # Read file content
        file_content = await file.read()

        # Extract text based on file type
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file_content)
        elif file_extension == 'docx':
            text = extract_text_from_docx(file_content)
        elif file_extension == 'txt':
            text = extract_text_from_txt(file_content)
        elif file_extension in ['xlsx', 'xls']:
            text = extract_text_from_excel(file_content)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Supported formats: PDF, DOCX, TXT, Excel"
            )

        # Prepare prompt for legal analysis
        prompt = f"""
        You are a highly skilled legal expert specializing in analyzing case details.
        Your task is to examine the provided information, identify the nature of the case, analyze it comprehensively,
        and propose potential legal defenses or solutions to resolve the issue or minimize liability. Always consider
        valid and logical arguments that comply with legal standards and ethical practices.
        Provide the analysis and solutions exclusively in Arabic.

        Case details:
        {text}

        Ensure that the response is:
        1. Comprehensive and clear.
        2. Accurate and logically sound.
        3. Aligned with legal norms and principles.
        4. Presented in a professional and concise manner.
        """

        # Get model response
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        analysis = completion.choices[0].message["content"].strip()

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return CaseAnalysisResponse(
            analysis=analysis,
            processing_time=processing_time,
            tokens_used=len(analysis.split()),
            confidence_score=0.9
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing case file: {str(e)}"
        )


# Add after other endpoints
@app.post("/lawyers/withdraw", response_model=WithdrawalRequestResponse)
def create_withdrawal_request(
    request: WithdrawalRequestCreate,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new withdrawal request"""
    if current_user.role != RoleEnum.lawyer:
        raise HTTPException(
            status_code=403,
            detail="Only lawyers can create withdrawal requests"
        )

    if current_user.lawyer_balance < 100:
        raise HTTPException(
            status_code=400,
            detail="Minimum withdrawal amount is 100 EGP"
        )

    if request.amount > current_user.lawyer_balance:
        raise HTTPException(
            status_code=400,
            detail="Withdrawal amount cannot exceed current balance"
        )

    if not request.phone_number or len(request.phone_number) < 10:
        raise HTTPException(
            status_code=400,
            detail="Invalid phone number"
        )

    # Create withdrawal request
    withdrawal_request = WithdrawalRequest(
        lawyer_id=current_user.user_id,
        amount=request.amount,
        phone_number=request.phone_number,
        status=WithdrawalStatusEnum.pending
    )

    db.add(withdrawal_request)
    db.commit()
    db.refresh(withdrawal_request)

    # Get lawyer name for response
    lawyer_name = f"{current_user.first_name} {current_user.last_name}"

    return WithdrawalRequestResponse(
        **withdrawal_request.__dict__,
        lawyer_name=lawyer_name
    )


@app.get("/admin/withdrawal-requests", response_model=List[WithdrawalRequestResponse])
def get_all_withdrawal_requests(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    status: Optional[WithdrawalStatusEnum] = None
):
    """Get all withdrawal requests (admin only)"""
    if current_user.role != RoleEnum.admin:
        raise HTTPException(
            status_code=403,
            detail="Only admin can access withdrawal requests"
        )

    query = db.query(WithdrawalRequest)
    if status:
        query = query.filter(WithdrawalRequest.status == status)

    requests = query.order_by(WithdrawalRequest.created_at.desc()).all()

    # Get lawyer names for response
    response = []
    for req in requests:
        lawyer = db.query(DBUser).filter(DBUser.user_id == req.lawyer_id).first()
        lawyer_name = f"{lawyer.first_name} {lawyer.last_name}" if lawyer else "Unknown"
        response.append(WithdrawalRequestResponse(
            **req.__dict__,
            lawyer_name=lawyer_name
        ))

    return response

@app.put("/admin/withdrawal-requests/{request_id}", response_model=WithdrawalRequestResponse)
def update_withdrawal_status(
    request_id: int,
    status_update: WithdrawalStatusUpdate,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update withdrawal request status (admin only)"""
    if current_user.role != RoleEnum.admin:
        raise HTTPException(
            status_code=403,
            detail="Only admin can update withdrawal requests"
        )

    request = db.query(WithdrawalRequest).filter(WithdrawalRequest.request_id == request_id).first()
    if not request:
        raise HTTPException(
            status_code=404,
            detail="Withdrawal request not found"
        )

    # Update request status
    request.status = status_update.status
    request.admin_notes = status_update.admin_notes
    request.updated_at = datetime.utcnow()

    # If request is paid, update lawyer's balance
    if status_update.status == WithdrawalStatusEnum.paid:
        lawyer = db.query(DBUser).filter(DBUser.user_id == request.lawyer_id).first()
        if lawyer:
            lawyer.lawyer_balance -= request.amount
            db.add(lawyer)

    db.add(request)
    db.commit()
    db.refresh(request)

    # Get lawyer name for response
    lawyer = db.query(DBUser).filter(DBUser.user_id == request.lawyer_id).first()
    lawyer_name = f"{lawyer.first_name} {lawyer.last_name}" if lawyer else "Unknown"

    return WithdrawalRequestResponse(
        **request.__dict__,
        lawyer_name=lawyer_name
    )

@app.get("/lawyers/withdrawal-requests", response_model=List[WithdrawalRequestResponse])
def get_lawyer_withdrawal_requests(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get lawyer's withdrawal requests"""
    if current_user.role != RoleEnum.lawyer:
        raise HTTPException(
            status_code=403,
            detail="Only lawyers can access their withdrawal requests"
        )

    requests = db.query(WithdrawalRequest).filter(
        WithdrawalRequest.lawyer_id == current_user.user_id
    ).order_by(WithdrawalRequest.created_at.desc()).all()

    lawyer_name = f"{current_user.first_name} {current_user.last_name}"

    return [
        WithdrawalRequestResponse(
            **req.__dict__,
            lawyer_name=lawyer_name
        )
        for req in requests
    ]




@app.get("/users/me/lawyer-chats", response_model=List[LawyerChatSummary])
async def get_user_lawyer_chats(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role == RoleEnum.lawyer:
        raise HTTPException(status_code=403, detail="Lawyers cannot access this endpoint.")

    chats = db.query(Chat).filter(
        Chat.chat_type == ChatTypeEnum.lawyer_client,
        Chat.status == ChatStatusEnum.active,
        Chat.participants.contains([current_user.user_id])
    ).all()

    result = []
    for chat in chats:
        last_query = db.query(Queries).filter(
            Queries.chat_id == chat.chat_id
        ).order_by(Queries.created_at.desc()).first()

        last_response = db.query(Responses).filter(
            Responses.chat_id == chat.chat_id
        ).order_by(Responses.created_at.desc()).first()

        last_message_data = None
        last_message_time = None
        if last_query and last_response:
            if last_query.created_at > last_response.created_at:
                last_message_data = last_query.query_text
                last_message_time = last_query.created_at
            else:
                last_message_data = last_response.responses_text
                last_message_time = last_response.created_at
        elif last_query:
            last_message_data = last_query.query_text
            last_message_time = last_query.created_at
        elif last_response:
            last_message_data = last_response.responses_text
            last_message_time = last_response.created_at
        else:
            last_message_data = "no messages yet"
            last_message_time = chat.started_at

        lawyer_id = None
        for participant_id in chat.participants:
            if participant_id != current_user.user_id:
                potential_lawyer = db.query(DBUser).filter(DBUser.user_id == participant_id).first()
                if potential_lawyer and potential_lawyer.role == RoleEnum.lawyer:
                    lawyer_id = potential_lawyer.user_id
                    break

        lawyer = None
        if lawyer_id:
            lawyer = db.query(DBUser).filter(DBUser.user_id == lawyer_id).first()

        if lawyer and last_message_data:
            unread_count = db.query(Responses).filter(
                Responses.chat_id == chat.chat_id,
                Responses.generated_by == f"lawyer_{lawyer.user_id}",
                Responses.status == "sent",
            ).count()


            result.append(LawyerChatSummary(
                chat_id=chat.chat_id,
                participant_id=lawyer.user_id,
                participant_name=f"{lawyer.first_name} {lawyer.last_name}",
                last_message=last_message_data,
                last_message_time=last_message_time,
                unread_count=unread_count,
                status=chat.status,
                created_at=chat.started_at
            ))

    return result

@app.get("/lawyers/me/client-chats", response_model=List[LawyerChatSummary])
async def get_lawyer_client_chats(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in [RoleEnum.lawyer, RoleEnum.admin]:
        raise HTTPException(status_code=403, detail="You are not authorized to access this endpoint.")

    chats = db.query(Chat).filter(
        Chat.chat_type == ChatTypeEnum.lawyer_client,
        Chat.status == ChatStatusEnum.active,
        Chat.participants.contains([current_user.user_id])
    ).all()

    result = []
    for chat in chats:
        last_query = db.query(Queries).filter(
            Queries.chat_id == chat.chat_id
        ).order_by(Queries.created_at.desc()).first()

        last_response = db.query(Responses).filter(
            Responses.chat_id == chat.chat_id
        ).order_by(Responses.created_at.desc()).first()

        last_message_data = None
        last_message_time = None
        if last_query and last_response:
            if last_query.created_at > last_response.created_at:
                last_message_data = last_query.query_text
                last_message_time = last_query.created_at
            else:
                last_message_data = last_response.responses_text
                last_message_time = last_response.created_at
        elif last_query:
            last_message_data = last_query.query_text
            last_message_time = last_query.created_at
        elif last_response:
            last_message_data = last_response.responses_text
            last_message_time = last_response.created_at
        else:
            last_message_data = "no messages yet"
            last_message_time = chat.started_at

        client_id = None
        for participant_id in chat.participants:
            if participant_id != current_user.user_id:
                potential_client = db.query(DBUser).filter(DBUser.user_id == participant_id).first()
                if potential_client and potential_client.role != RoleEnum.lawyer:
                    client_id = potential_client.user_id
                    break

        client = None
        if client_id:
            client = db.query(DBUser).filter(DBUser.user_id == client_id).first()

        if client:
            unread_count = db.query(Queries).filter(
                Queries.chat_id == chat.chat_id,
                Queries.status == "sent",
                Queries.user_id == client.user_id
            ).count()

            result.append(LawyerChatSummary(
                chat_id=chat.chat_id,
                participant_id=client.user_id,
                participant_name=f"{client.first_name} {client.last_name}",
                last_message=last_message_data,
                last_message_time=last_message_time,
                unread_count=unread_count,
                status=chat.status,
                created_at=chat.started_at
            ))

    return result


if __name__ == "__main__":

    # print("API is running locally at: http://127.0.0.1:8000")
    # uvicorn.run(app, host="127.0.0.1", port=7000)
    ngrok.set_auth_token("2wEj5nPnrkSM8muaIx8iizZ9dnk_J6KucNuWHvcAaKobsxwx")
    public_url = ngrok.connect(addr=8000, domain="enabled-early-vulture.ngrok-free.app")
    print("API is live at:", public_url)
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Authentication routes for user login, registration, and token management.
"""

from typing import Annotated
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas import (
    TokenResponse, 
    UserResponse, 
    UserCreate, 
    UserLogin,
    APIResponse
)
from ..auth import (
    authenticate_user, 
    create_access_token, 
    get_current_user,
    verify_token,
    get_password_hash
)
from ..models import User

# Security scheme
security = HTTPBearer()

# Router
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=APIResponse[TokenResponse], summary="User Login")
async def login_for_access_token(
    form_data: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return access token.
    
    **Parameters:**
    - **username**: User's username or email
    - **password**: User's password
    
    **Returns:**
    - **access_token**: JWT token for API access
    - **token_type**: Always "bearer"
    - **expires_in**: Token expiration time in seconds
    
    **Example:**
    ```json
    {
        "success": true,
        "data": {
            "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
            "token_type": "bearer",
            "expires_in": 3600
        },
        "message": "Login successful"
    }
    ```
    """
    # Authenticate user
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user account",
        )
    
    # Create token
    access_token_expires = timedelta(hours=24)  # 24 hours
    access_token = create_access_token(
        data={"sub": user.username, "user_id": str(user.id)},
        expires_delta=access_token_expires
    )
    
    token_data = TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds())
    )
    
    return APIResponse(
        success=True,
        data=token_data,
        message="Login successful"
    )


@router.post("/register", response_model=APIResponse[UserResponse], summary="User Registration")
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user account.
    
    **Note:** Registration creates regular users. Admin users must be created manually.
    
    **Parameters:**
    - **username**: Unique username (3-50 characters)
    - **email**: Valid email address
    - **password**: Password (minimum 8 characters)
    - **full_name**: User's full name
    
    **Returns:**
    - **User information** (excluding password)
    
    **Example:**
    ```json
    {
        "success": true,
        "data": {
            "id": 1,
            "username": "johndoe",
            "email": "john@example.com",
            "full_name": "John Doe",
            "is_active": true,
            "is_admin": false,
            "created_at": "2024-01-15T10:30:00Z"
        },
        "message": "User registered successfully"
    }
    ```
    """
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=hashed_password,
        is_active=True,
        is_admin=False  # Regular users are not admin by default
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Convert to response model
    user_response = UserResponse(
        id=new_user.id,
        username=new_user.username,
        email=new_user.email,
        full_name=new_user.full_name,
        is_active=new_user.is_active,
        is_admin=new_user.is_admin,
        created_at=new_user.created_at
    )
    
    return APIResponse(
        success=True,
        data=user_response,
        message="User registered successfully"
    )


@router.get("/me", response_model=APIResponse[UserResponse], summary="Get Current User")
async def read_users_me(
    current_user: User = Depends(get_current_user)
):
    """
    Get current authenticated user information.
    
    **Authentication:** Bearer token required
    
    **Returns:**
    - **Current user information**
    
    **Example:**
    ```json
    {
        "success": true,
        "data": {
            "id": 1,
            "username": "johndoe",
            "email": "john@example.com",
            "full_name": "John Doe",
            "is_active": true,
            "is_admin": false,
            "created_at": "2024-01-15T10:30:00Z"
        },
        "message": "Current user information"
    }
    ```
    """
    user_response = UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_admin=current_user.is_admin,
        created_at=current_user.created_at
    )
    
    return APIResponse(
        success=True,
        data=user_response,
        message="Current user information"
    )


@router.post("/verify-token", response_model=APIResponse[dict], summary="Verify Token")
async def verify_access_token(
    token: str = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Verify if an access token is valid.
    
    **Authentication:** Bearer token required
    
    **Returns:**
    - **Token validity status**
    - **User information** if token is valid
    
    **Example:**
    ```json
    {
        "success": true,
        "data": {
            "valid": true,
            "username": "johndoe",
            "user_id": "1",
            "expires_at": "2024-01-16T10:30:00Z"
        },
        "message": "Token is valid"
    }
    ```
    """
    try:
        # Extract token from bearer format
        if hasattr(token, 'credentials'):
            token_str = token.credentials
        else:
            token_str = str(token)
        
        # Verify token
        payload = verify_token(token_str)
        if not payload:
            return APIResponse(
                success=False,
                data={"valid": False},
                message="Invalid token"
            )
        
        # Get user information
        username = payload.get("sub")
        user_id = payload.get("user_id")
        exp = payload.get("exp")
        
        # Convert expiration to datetime
        from datetime import datetime
        expires_at = datetime.fromtimestamp(exp) if exp else None
        
        return APIResponse(
            success=True,
            data={
                "valid": True,
                "username": username,
                "user_id": user_id,
                "expires_at": expires_at
            },
            message="Token is valid"
        )
    
    except Exception:
        return APIResponse(
            success=False,
            data={"valid": False},
            message="Invalid token"
        )


@router.post("/refresh", response_model=APIResponse[TokenResponse], summary="Refresh Token")
async def refresh_access_token(
    current_user: User = Depends(get_current_user)
):
    """
    Refresh access token for current user.
    
    **Authentication:** Valid bearer token required
    
    **Returns:**
    - **New access token**
    
    **Example:**
    ```json
    {
        "success": true,
        "data": {
            "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
            "token_type": "bearer",
            "expires_in": 3600
        },
        "message": "Token refreshed successfully"
    }
    ```
    """
    # Create new token
    access_token_expires = timedelta(hours=24)
    access_token = create_access_token(
        data={"sub": current_user.username, "user_id": str(current_user.id)},
        expires_delta=access_token_expires
    )
    
    token_data = TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds())
    )
    
    return APIResponse(
        success=True,
        data=token_data,
        message="Token refreshed successfully"
    )
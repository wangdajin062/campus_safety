package com.campus.safety.model;

public class ApiResponse<T> {
    public int    code;
    public T      data;
    public String message;

    public boolean isSuccess() {
        return code == 200;
    }
}

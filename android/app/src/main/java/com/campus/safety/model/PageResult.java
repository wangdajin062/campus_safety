package com.campus.safety.model;

import java.util.List;

public class PageResult<T> {
    public int total;
    public int page;
    public int limit;
    public List<T> items;
}

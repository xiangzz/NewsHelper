/**
 * 
 */
package com.xiang.topicmodel.framework;

import java.util.ArrayList;

/**
 * @author rivercrab3
 *
 */
public abstract class DataProcessor {
	
	/**
	 * 
	 */
	public Object data;
	public Filter dataFilter;
	
	/**
	 * 读入数据，保存到data中
	 * @param source
	 */
	public abstract void read();
	
	
	/**
	 * 读入filter，作为处理文本的过滤器
	 * @param filter
	 */
	public abstract void setFilter(Filter filter);
	
	
	/**
	 * 处理文本，输出单词list
	 */
	public abstract void process();
	
}

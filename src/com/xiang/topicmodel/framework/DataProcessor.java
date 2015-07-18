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
	 * �������ݣ����浽data��
	 * @param source
	 */
	public abstract void read();
	
	
	/**
	 * ����filter����Ϊ�����ı��Ĺ�����
	 * @param filter
	 */
	public abstract void setFilter(Filter filter);
	
	
	/**
	 * �����ı����������list
	 */
	public abstract void process();
	
}

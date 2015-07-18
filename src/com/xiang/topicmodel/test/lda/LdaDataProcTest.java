package com.xiang.topicmodel.test.lda;

import java.util.ArrayList;

import com.xiang.topicmodel.lda.LdaDataProc;
import com.xiang.topicmodel.lda.LdaFilter;
import com.xiang.topicmodel.lda.LdaFilterRule;

public class LdaDataProcTest {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		LdaFilterRule rule = new LdaFilterRule("./data/stopWords");
		LdaFilter filter = new LdaFilter(rule);
		LdaDataProc dp = new LdaDataProc("./data/news", filter);
		dp.read();
		dp.process();
		ArrayList< ArrayList<String> > d = (ArrayList< ArrayList<String> >) dp.getData();
		for(ArrayList<String> al:d){
			System.out.println("-------------------");
			for(String s:al){
				System.out.print(s+",");
			}
			System.out.println();
		}

	}

}

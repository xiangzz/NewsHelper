package com.xiang.topicmodel.lda;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.Map.Entry;

import liuyang.nlp.lda.com.FileUtil;
import liuyang.nlp.lda.conf.PathConfig;
import liuyang.nlp.lda.main.Document;

import com.xiang.topicmodel.framework.DataProcessor;
import com.xiang.topicmodel.framework.Filter;

public class LdaDataProc extends DataProcessor {

	public String fileDir;
	public ArrayList<ArrayList<String>> docs;
	public Map<String, Integer> termToIndexMap;
	public ArrayList<String> indexToTermMap;
	public Map<String,Integer> termCountMap;
	
	public LdaDataProc(String fileDir,Filter filter){
		this.fileDir = fileDir;
		this.dataFilter = filter;
		this.docs = new ArrayList<ArrayList<String>>();
		this.termToIndexMap = new HashMap<String, Integer>();
		this.indexToTermMap = new ArrayList<String>();
	}
	
	
	private ArrayList<String> readFile(File file){
		BufferedReader reader = null;
		ArrayList<String> words = new ArrayList<String>();
		try {
			reader = new BufferedReader(new FileReader(file));
			String line = null;
			while ((line = reader.readLine()) != null) {
				StringTokenizer strTok = new StringTokenizer(line);
				while (strTok.hasMoreTokens()) {
					String token = strTok.nextToken();
					words.add(token.toLowerCase().trim());
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return words;
	}
	
	
	@Override
	public void read() {
		// TODO Auto-generated method stub
		File dir = new File(this.fileDir);
		//如果输入的是目录,这意味着训练
		if(dir.isDirectory()){
			for(File docFile : dir.listFiles()){
				ArrayList<String> words = this.readFile(docFile);
				this.docs.add(words);
			}
		}
		//如果输入的是文件，这意味着估计
		if(dir.isFile()){
			ArrayList<String> words = this.readFile(dir.getAbsoluteFile());
			this.docs.add(words);
		}
	}

	@Override
	public void setFilter(Filter filter) {
		// TODO Auto-generated method stub
		this.dataFilter = filter;
	}

	@Override
	public void process() {
		// TODO Auto-generated method stub
		//先过滤
		this.dataFilter.filter(this.docs);
		//再统计
		for(ArrayList<String> doc:(ArrayList<ArrayList<String>>)this.docs){
			for(String word: doc){
				if(!this.termToIndexMap.containsKey(word)){
					Integer newIndex = this.termToIndexMap.size();
					this.termToIndexMap.put(word, newIndex);
					this.indexToTermMap.add(word);
				}
			}
		}
		//赋值
		this.data = this.docs;
	}
	
	public ArrayList getData(){
		return (ArrayList) this.data;
	}
	//added by tokyo
	/**
	 * 保存termToInDexMap
	 */
	public void savetermToIndexMap(String filePath){

		String fileName = "termToIndexMap";
		ArrayList<String> lines = new ArrayList<String>();
		for (Entry<String, Integer> termToIndex : termToIndexMap.entrySet()) {
			String line = termToIndex.getKey()+":"+termToIndex.getValue();
			lines.add(line);
		}
		FileUtil.writeLines(filePath + fileName, lines);
	}	
	
}

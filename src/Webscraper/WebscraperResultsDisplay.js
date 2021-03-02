import React, { Component } from 'react';

export default class WebscraperResultsDisplay extends Component {
    constructor(props){
        super(props);
        this.state = {
            sorted: false,
            show_sentence: false,
            shown_sentence: ''
        }
        this.handleSort = this.handleSort.bind(this);
        this.sentenceClickHandler = this.sentenceClickHandler.bind(this);
    }
    sentenceClickHandler = sentence => {
        this.setState({
            show_sentence: !this.state.show_sentence,
            shown_sentence: sentence.order
        })
    }
    handleWord = word => {
        //const word_and_score = word.split(":");
        var the_word = word[0];
        const score_float = parseFloat(word[1]);
        the_word = the_word.replace("##", "");
        return(
            <span className='word-wrapper'>
                <p>{the_word}</p>
                <p style={{color: (score_float > .6) ? "red" : "green"}}>{Math.round(score_float*1000)/1000}</p>
            </span>
        )
    }
    showSentence = sentence => {
        if(this.state.show_sentence && this.state.shown_sentence === sentence.order){
            return sentence.words.map ((word) => {
                return (
                    <div>{this.handleWord(word)}</div>  
                )
            })
        }
        else {
            return(
                <div></div>
            )
        }
    }
    getArticleContent() {
        if (this.props.output.title !== ""){
            var sentences = this.props.output.bias_results.sentence_results;
            if(this.state.sorted === true){
                sentences = this.props.output.bias_results.sentence_results.sort((a, b) => (a.bias_score > b.bias_score)? -1 : 1);
            } else {
                sentences = this.props.output.bias_results.sentence_results.sort((a,b) => (a.order > b.order) ? 1 : -1)
            }
            return sentences.map((sentence) => {
                return (
                    <div className='sentence-wrapper'>
                        <div className='individual-sentence-wrapper' onClick={() => this.sentenceClickHandler(sentence)}>
                            <p className='sentence-score' style={{background: (sentence.bias_score > 6) ? "rgb(255, 208, 208)" : "rgb(208, 255, 210)"}} name='average'>{Math.round(sentence.bias_score * 100)/100}</p>
                            <p name='most-biased'>{sentence.max_biased_word.replace('##', '')}</p>
                        </div>
                            <span className='expanded-sentence-wrapper'>{this.showSentence(sentence)}</span>
                    </div> 
                );
            });
        } else {
            return (
                <div></div>
            )
        }
    }
    handleSort(){
        this.setState({sorted: !this.state.sorted});
    }
    render() {
        return(
            <div>
                <ul className='webscraper-results-list'>
                    <span className='webscraper-result-wrapper'>
                        <label className='webscraper-result-label' htmlFor='title'>Title</label>
                        <li name='title' className='webscraper-result'>{this.props.output.title}</li>
                    </span>
                    <span className='webscraper-result-wrapper'>
                        <label className='webscraper-result-label' htmlFor='author'>Author</label>
                        <li name='author' className='webscraper-result'>{this.props.output.author}</li>
                    </span>
                    <span className='score-runtime-wrapper'>
                        <span className='webscraper-result-wrapper'>
                            <label className='webscraper-result-label' htmlFor='article-score'>Article Score</label>
                            <li style={{color: (this.props.output.bias_results.article_score > 6) ? "red" : "green"}} name='article-score' className='webscraper-article-numbers'><strong>{Math.round(this.props.output.bias_results.article_score * 100)/100}</strong></li>
                        </span>
                        <span className='webscraper-result-wrapper'>
                            <label className='webscraper-result-label' htmlFor='runtime'>Runtime</label>
                            <li name='runtime' className='webscraper-result'>{this.props.output.bias_results.runtime.substr(0,5)} seconds</li>
                        </span>
                    </span>
                    <span className='webscraper-result-wrapper'>
                        <label className='webscraper-result-label' htmlFor='related-articles'>Related Articles</label>
                        <span name='related-articles' className='webscraper-result'>
                            <p className='related-article-link'>{this.props.output.related.partisan_left}</p>
                            <p className='related-article-link'>{this.props.output.related.skews_left}</p>
                            <p className='related-article-link'>{this.props.output.related.middle}</p>
                            <p className='related-article-link'>{this.props.output.related.skews_right}</p>
                            <p className='related-article-link'>{this.props.output.related.partisan_right}</p>
                        </span>
                    </span>
                    <span className='webscraper-result-wrapper'>
                        <label className='webscraper-result-label' htmlFor='article'>Sentence Level Scores</label>
                        <ul className='webscraper-article-content-list' name='article'>
                            <span className='sort-wrapper'>
                                <button className='sort-button' onClick={this.handleSort}>Toggle Sort</button>
                                <p className='sorting-by'>Currently sorting by <strong>{(this.state.sorted) ? "bias score" : "order of appearance"}</strong></p>
                            </span>
                            <span className='sentences-header'>
                                <p>Sentence Score</p>
                                <p>Most Biased Word</p>
                            </span>
                            {this.getArticleContent()}
                        </ul>
                    </span>
                </ul>
            </div>
        )
    }
}
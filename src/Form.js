import axios from 'axios';
import React from 'react';
import './Form.css';

class Form extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            text: "Enter text here",
            words: [],
            averageBias: [],
            biasedWord: []
        };

        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }
    handleChange(event){
        this.setState({text: event.target.value}); //, response: event.target.response});
    }
    handleSubmit(event){
        event.preventDefault();
        const myText = this.state.text;
        console.log("Input: " + myText);
        axios.post("/api/results", {myText})
            .then(res => {
                this.setState({ words: [...this.state.words, res.data] })
                this.setState({ averageBias: [...this.state.averageBias, res.data.average] })
                this.setState({ biasedWord: [...this.state.biasedWord, res.data.max_biased_word] })
                console.log(res.data);
            });
    }
    render(){
        var wordList = this.state.words.map((item) =>
            <section className="results_display">
                <ul id="results">
                        <li className="word" key={item}>
                            {item.words.map((word) =>
                                <ul className="list-group list-group-flush">
                                    <li className="list-group-item" key={word}>
                                        {word}
                                    </li>
                                </ul>
                            )}
                        </li>
                        <li className="other_info" key={item.average}>
                            <p>
                                Average Score: <strong>{item.average}</strong>.
                            </p>
                            <p>
                                The most biased word is: <strong>{item.max_biased_word}</strong>.
                            </p>
                        </li>
                </ul>
            </section>
            )
        return(
        <div>
            <form className="form" onSubmit={this.handleSubmit}>
            <p></p>
                <label>
                    <strong>Text:</strong> <input className="text_input" key="text-input" type="text" value={this.state.text} onChange={this.handleChange} />
                </label>
                <button className="submit_button" type="submit" value="Submit">Submit</button>
                <ul className="results_wrapper">{wordList}</ul>
            </form>
            <p></p>
        </div>
        );
    }
}
export default Form;